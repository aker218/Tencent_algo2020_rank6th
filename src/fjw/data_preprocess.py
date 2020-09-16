#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import _pickle as pk
import os
from IPython.display import display,HTML
from category_encoders import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from transformers import *
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence
np.random.seed(13)
import collections
import time
import logging
import multiprocessing
import copy
logger = logging.getLogger(__name__)
from collections import defaultdict
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO 
)
ARG=collections.namedtuple('ARG',['train_batch_size',
 'eval_batch_size',
 'weight_decay',
 'learning_rate',
 'adam_epsilon',
 'num_train_epochs',
 'warmup_steps',
 'gradient_accumulation_steps',
 'save_steps',
 'max_grad_norm',
 'model_name_or_path',
 'output_dir',
 'seed',
 'device',
 'n_gpu',
 'max_steps',
 'output_mode',
'fp16_opt_level',
'fp16',
'card_list'])


# In[138]:


test_num=1  #用于测试的比例，使用全部数据则test_num为1，使用 1/1000为1000


# ## Target Encoding

# In[2]:


train_click_log=pd.read_csv("../../var/data/train_preliminary/click_log.csv")
train_usr_label=pd.read_csv("../../var/data/train_preliminary/user.csv")
train_ad_info=pd.read_csv("../../var/data/train_preliminary/ad.csv")
new_train_click_log=pd.read_csv("../../var/data/train_semi_final/click_log.csv")
new_train_usr_label=pd.read_csv("../../var/data/train_semi_final/user.csv")
new_train_ad_info=pd.read_csv("../../var/data/train_semi_final/ad.csv")


# In[3]:


train_usr_label=pd.concat([train_usr_label,new_train_usr_label]).drop_duplicates()
train_click_log=pd.concat([train_click_log,new_train_click_log]).drop_duplicates()
train_ad_info=pd.concat([train_ad_info,new_train_ad_info]).drop_duplicates()


# In[4]:


test_click_log=pd.read_csv("../../var/data/test/click_log.csv")
test_ad_info=pd.read_csv("../../var/data/test/ad.csv")


# In[5]:


cid_list=train_click_log['creative_id'].to_list()
train_full_info=pd.merge(train_click_log,train_ad_info.query('creative_id==@cid_list'),how='left')
train_full_info=pd.merge(train_full_info,train_usr_label,how='left')
train_full_info=pd.get_dummies(train_full_info,columns=['age','gender'])


# In[6]:



cid_list=test_click_log['creative_id'].to_list()
test_full_info=pd.merge(test_click_log,test_ad_info.query('creative_id==@cid_list'),how='left')


# In[8]:


cols=['click_times','time','creative_id', 'ad_id','product_id','product_category','advertiser_id','industry'] 
for col in cols:
    print(col)
    pk.dump(train_full_info[["user_id"]+[col]+['age_'+str(i) for i in range(1,11)]            +['gender_'+str(i) for i in range(1,3)]],            open("../../var/fjw/train_mid/"+col+"_full_info.pk","wb"))
cols=['click_times','time','creative_id', 'ad_id','product_id','product_category','advertiser_id','industry'] 
for col in cols:
    print(col)
    pk.dump(test_full_info[["user_id"]+[col]],            open("../../var/fjw/test_mid/"+col+"_full_info.pk","wb"))


# In[9]:


cols=['click_times','time','creative_id', 'ad_id','product_id','product_category','advertiser_id','industry']
map_dicts=[dict() for i in range(6)]
usr_id_dict=dict()
for col in cols:
    print("+++++++++++++++++"+col)
    train_full_info=pk.load(open("../../var/fjw/train_mid/"+col+"_full_info.pk","rb"))
    test_full_info=pk.load(open("../../var/fjw/test_mid/"+col+"_full_info.pk","rb"))
#     if col=='creative_id':
#         break
    train_id_info=train_full_info['user_id'].drop_duplicates()
    n_folds=5
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=20)
    targets=train_full_info.columns[-12:].tolist()
    for target in targets:
        train_full_info.insert(train_full_info.shape[1],col+"_"+target,0)
    col_idx=dict([(col,idx) for idx,col in enumerate(train_full_info.columns)])
    train_full_info=train_full_info
    temp_cols=[col]
    for fold_,(train_idx,val_idx) in tqdm(enumerate(kf.split(train_id_info))):
        print("第",fold_+1,"折")
        X_train=train_full_info[train_full_info['user_id'].isin(train_id_info.iloc[train_idx])].loc[:,temp_cols+targets]
        X_test=train_full_info[train_full_info['user_id'].isin(train_id_info.iloc[val_idx])].loc[:,temp_cols+targets]
        enc = TargetEncoder(cols=temp_cols)
        temp_datasets=[]
        real_val_idx=train_full_info[train_full_info['user_id'].isin(train_id_info.iloc[val_idx])].index
        for target in targets:
            print("encoding...",target)
            y_train=X_train[target]
            enc.fit(X_train, y_train)
            testing_numeric_dataset = enc.transform(X_test)
            testing_numeric_dataset.columns=[col+"_"+target if col in temp_cols else col for col in testing_numeric_dataset.columns]
            print("end...")
            train_full_info.iloc[real_val_idx,[col_idx[col+"_"+target] for col in temp_cols]]=            testing_numeric_dataset.loc[:,[col+"_"+target for col in temp_cols]]
    temp_test_full_info=test_full_info
    temp_cols=[col]
    for target in targets:
        default_mean=train_full_info[col+'_'+target].mean()
        map_dict=dict(train_full_info[[col,col+'_'+target]].groupby(col).mean()[col+'_'+target])
        test_cols=list(set(temp_test_full_info[col].tolist()))
        nums=0
        for e in tqdm(test_cols):
            if e not in map_dict.keys():
                map_dict[e]=default_mean
                nums+=1
        map_info=[]
        for key,value in map_dict.items():
            map_info.append([key,value])
        print(col," ",target)
#         print(len(test_cols))
#         print(nums)
        map_df=pd.DataFrame(map_info,columns=[col,col+"_"+target])
        temp_test_full_info=pd.merge(temp_test_full_info,map_df,how='left')
    n_folds=5
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=20)
    targets=["age_"+str(i) for i in range(1,11)]+["gender_"+str(i) for i in range(1,3)]
    target_cols=train_full_info.columns[-12:].tolist()

    id_lists=[]
    for fold_,(train_idx,val_idx) in tqdm(enumerate(kf.split(train_id_info))):
        id_list=train_id_info.iloc[val_idx].to_list()
        id_lists.append(id_list)
        val_info=train_full_info[train_full_info['user_id'].isin(train_id_info.iloc[val_idx])].loc[:,temp_cols+target_cols]
        for e in id_list:
            usr_id_dict[e]=fold_
        for col in tqdm(temp_cols):
            temp=val_info.loc[:,[col]+[col+"_"+target for target in targets]].drop_duplicates().set_index(col)
            temp=pd.Series(data=[e for e in temp.values],index=temp.index)
            if col in ['creative_id', 'ad_id','product_id','product_category','advertiser_id','industry']:
                temp.index=temp.index.astype('str')
            if col in ['product_id','industry']:
                if '\\N' in temp.index:
                    temp['0']=temp['\\N']
            map_dicts[fold_][col]=temp
    # pk.dump([usr_id_dict,map_dicts],open("./dataset/train_mid/target_info.pk","wb"))
    test_full_info=temp_test_full_info
    test_id_info=test_full_info['user_id'].drop_duplicates()
    id_list=test_id_info.to_list()
    for e in id_list:
        usr_id_dict[e]=5
    targets=["age_"+str(i) for i in range(1,11)]+["gender_"+str(i) for i in range(1,3)]
    target_cols=test_full_info.columns[-12:].tolist()
    val_info=test_full_info.loc[:,temp_cols+target_cols]
    for col in tqdm(temp_cols):
        temp=val_info.loc[:,[col]+[col+"_"+target for target in targets]].drop_duplicates().set_index(col)
        temp=pd.Series(data=[e for e in temp.values],index=temp.index)
        if col in ['creative_id', 'ad_id','product_id','product_category','advertiser_id','industry']:
            temp.index=temp.index.astype('str')
        if col in ['product_id','industry']:
            if '\\N' in temp.index:
                temp['0']=temp['\\N']
        map_dicts[5][col]=temp
pk.dump([usr_id_dict,map_dicts],open("../../var/fjw/train_mid/target_info.pk","wb"))
indexs=[]
values=[]
for idx in map_dicts[5]['industry'].index.drop_duplicates():
    indexs.append(idx)
    if len(map_dicts[5]['industry'][idx])>1 and len(map_dicts[5]['industry'][idx])!=12:
        assert  (map_dicts[5]['industry'][idx].iloc[0]==map_dicts[5]['industry'][idx].iloc[1]).all()
        values.append(map_dicts[5]['industry'][idx].iloc[0])
    else:
        values.append(map_dicts[5]['industry'][idx])
new_series=pd.Series(data=values,index=indexs)
map_dicts[5]['industry']=new_series
pk.dump([usr_id_dict,map_dicts],open("../../var/fjw/train_mid/target_info.pk","wb"))


# ## 信息读取

# In[2]:


train_ad_info=pd.read_csv("/home/huangweilin/fjw/competition/tencent_ad/dataset/train_preliminary/ad.csv")
new_train_ad_info=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_semi_final/ad.csv")
test_ad_info=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/test/ad.csv")
ad_info=pd.concat([train_ad_info,new_train_ad_info,test_ad_info]).drop_duplicates()
ad_info.to_csv("../../var/fjw/full_ad.csv",index=False)


# In[3]:


train_click_log=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_preliminary/click_log.csv")
train_usr_label=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_preliminary/user.csv")
train_ad_info=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_preliminary/ad.csv")


# In[4]:


new_train_click_log=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_semi_final/click_log.csv")
new_train_usr_label=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_semi_final/user.csv")
new_train_ad_info=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/train_semi_final/ad.csv")


# In[5]:


train_usr_label=pd.concat([train_usr_label,new_train_usr_label]).drop_duplicates()
train_click_log=pd.concat([train_click_log,new_train_click_log]).drop_duplicates()
train_ad_info=pd.concat([train_ad_info,new_train_ad_info]).drop_duplicates()


# In[6]:


test_click_log=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/test/click_log.csv")
test_ad_info=pd.read_csv("/home/huangweilin/hyr/competitions/algo_round2/data/test/ad.csv")


# ## user wise

# ### load

# In[7]:


cid_list=train_click_log['creative_id'].to_list()
train_full_info=pd.merge(train_click_log,train_ad_info.query('creative_id==@cid_list'),how='left')
train_full_info=pd.merge(train_full_info,train_usr_label,how='left')


# In[8]:


cid_list=test_click_log['creative_id'].to_list()
test_full_info=pd.merge(test_click_log,test_ad_info.query('creative_id==@cid_list'),how='left')


# In[2]:


if not os.path.exists("../../var/fjw/train_mid/usr_click_log_df.pk"):
    train_usr_click_log=[]
    for idx,info in tqdm(train_full_info.groupby('user_id')):
        t=info.sort_values('time').reset_index(drop=True)
        train_usr_click_log.append(t)
    columns=train_usr_click_log[0].columns
    new_info=[e.values.transpose().tolist() for e in train_usr_click_log[:]]
    new_train_info=pd.DataFrame(new_info,columns=columns)
    for idx in tqdm(range(new_train_info.shape[0])):
        new_train_info.iloc[idx]['product_id']=[int(e) for e in " ".join(new_train_info.iloc[idx]['product_id']).replace("\\N","0").split(" ")]
        new_train_info.iloc[idx]['industry']=[int(e) for e in " ".join(new_train_info.iloc[idx]['industry']).replace("\\N","0").split(" ")]
    pk.dump(new_train_info,open("../../var/fjw/train_mid/usr_click_log_df.pk","wb"))
    new_train_usr_click_log=new_train_info
else:
    print("loading train usr click log df...")
    new_train_usr_click_log=pk.load(open("../../var/fjw/train_mid/usr_click_log_df.pk","rb"))


# In[3]:


if not os.path.exists("../../var/fjw/test_mid/usr_click_log_df.pk"):
    test_usr_click_log=[]
    for idx,info in tqdm(test_full_info.groupby('user_id')):
        t=info.sort_values('time').reset_index(drop=True)
        test_usr_click_log.append(t)
    columns=test_usr_click_log[0].columns
    new_info=[e.values.transpose().tolist() for e in test_usr_click_log[:]]
    new_test_info=pd.DataFrame(new_info,columns=columns)
    for idx in tqdm(range(new_test_info.shape[0])):
        new_test_info.iloc[idx]['product_id']=[int(e) for e in " ".join(new_test_info.iloc[idx]['product_id']).replace("\\N","0").split(" ")]
        new_test_info.iloc[idx]['industry']=[int(e) for e in " ".join(new_test_info.iloc[idx]['industry']).replace("\\N","0").split(" ")]
    pk.dump(new_test_info,open("../../var/fjw/test_mid/usr_click_log_df.pk","wb"))
    new_test_usr_click_log=new_test_info
else:
    print("loading test usr click log df...")
    new_test_usr_click_log=pk.load(open("../../var/fjw/test_mid/usr_click_log_df.pk","rb"))


# ### usr seq

# In[4]:


full_info=pd.concat([new_train_usr_click_log.iloc[:,:-2],new_test_usr_click_log])


# In[22]:


cross_cols=['product_id','product_category','advertiser_id','industry']
usr_idxs=[e[0] for e in full_info['user_id'].values]
for idx in range(len(cross_cols)):
    for jdx in range(idx+1,len(cross_cols)):
        print(cross_cols[idx]+"_"+cross_cols[jdx])
        new_data=[]
        for a,b in tqdm(zip(full_info[cross_cols[idx]].values,full_info[cross_cols[jdx]].values)):
            new_data.append([str(ai)+"_"+str(bi) for ai,bi in zip(a,b)])
        new_series=pd.Series(data=new_data,index=usr_idxs)
        pk.dump(new_series,open("../../var/fjw/usr_seq/se_user_"+cross_cols[idx]+"_"+cross_cols[jdx]+".pickle","wb"))


# In[5]:



# cat_cols=['time','click_times','creative_id','ad_id','product_id','product_category','advertiser_id','industry']
cat_cols=['ad_id']
usr_idxs=[e[0] for e in full_info['user_id'].values]
for col in cat_cols:
    print(col)
    new_series=pd.Series(data=full_info[col].values,index=usr_idxs)
    pk.dump(new_series,open("../../var/fjw/usr_seq/se_user_"+col+".pickle","wb"))


# #### shuffle

# In[63]:


df_user_info = pk.load(open('../../var/hyr/df_user_info.pickle', 'rb'))


# In[64]:


def get_same_day(time_seq):
    s = None
    e = None
    ret = []
    for i in range(len(time_seq) - 1):
        if time_seq[i] == time_seq[i+1] and s is None:
            s = i
        if time_seq[i] != time_seq[i+1] and s is not None:
            e = i+1
            ret.append([s, e])
            s = None
            e = None
    return ret


# In[68]:


for i in tqdm(range(1, 4000001)):
    if i not in df_user_info.index:
        continue
    time_seq = df_user_info.at[i, 'time']
    creative_seq = np.array(df_user_info.at[i, 'creative_id']).astype('int32')
    click_time_seq = np.array(df_user_info.at[i, 'click_time']).astype('int32')
    same_days = get_same_day(time_seq)
    idx = np.arange(click_time_seq.shape[0]).astype('int32')
    for day in same_days:
        np.random.shuffle(idx[day[0]:day[1]])
    creative_seq = creative_seq[idx]
    click_time_seq = click_time_seq[idx]
    df_user_info.at[i, 'creative_id'] = creative_seq.tolist()
    df_user_info.at[i, 'click_time'] = click_time_seq.tolist()   


# In[70]:


info=pk.load(open("../../var/fjw/usr_seq/se_user_click_times.pickle","rb"))


# In[72]:


pk.dump(df_user_info['click_time'], open('../../var/fjw/usr_seq/se_user_click_time_shuffle.pk', 'wb'))
pk.dump(df_user_info['creative_id'], open('../../var/fjw/usr_seq/se_user_creative_id_shuffle.pk', 'wb'))


# In[73]:


pk.dump(df_user_info, open('../../var/hyr/df_user_info_shuffle.pickle', 'wb'))


# ### 统计特征聚合

# In[26]:


def freq(df):
    return df.value_counts().values[0]
def aggregate_features(df_, prefix):

    df = df_.copy()
    categorical_cols=['creative_id', 'ad_id','product_id','product_category','advertiser_id','industry']
    categorical_cols_func=dict([ (col,['nunique',freq]) for idx,col in enumerate(categorical_cols)])
    numeric_cols=['click_times','time']
    numeric_cols_func=dict([(col,['mean','max','min','std',"nunique",freq,"count"]) if idx==0 else (col,['mean','max','min','std',"nunique",freq]) for idx,col in enumerate(numeric_cols)])
    agg_func=dict()
    agg_func.update(numeric_cols_func)
    agg_func.update(categorical_cols_func)
    agg_df = df.groupby(['user_id']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)
    info=df.groupby(['user_id','time']).sum().groupby("user_id").agg({"click_times":['mean','max','min','std',"nunique",freq]})
    info.columns = [prefix+ '_'.join(col).strip() +"/times" for col in info.columns.values]
    info.reset_index(drop=False,inplace=True)
    agg_df=pd.merge(agg_df,info)
    
    return agg_df


# In[27]:


print("start!!!!")
train_click_log=pd.read_csv("../../var/data/train_preliminary/click_log.csv")
train_usr_label=pd.read_csv("../../var/data/train_preliminary/user.csv")
train_ad_info=pd.read_csv("../../var/data/train_preliminary/ad.csv")
new_train_click_log=pd.read_csv("../../var/data/train_semi_final/click_log.csv")
new_train_usr_label=pd.read_csv("../../var/data/train_semi_final/user.csv")
new_train_ad_info=pd.read_csv("../../var/data/train_semi_final/ad.csv")
train_usr_label=pd.concat([train_usr_label,new_train_usr_label]).drop_duplicates()
train_click_log=pd.concat([train_click_log,new_train_click_log]).drop_duplicates()
train_ad_info=pd.concat([train_ad_info,new_train_ad_info]).drop_duplicates()
test_click_log=pd.read_csv("../../var/data/test/click_log.csv")
test_ad_info=pd.read_csv("../../var/data/test/ad.csv")
cid_list=train_click_log['creative_id'].to_list()
train_full_info=pd.merge(train_click_log,train_ad_info.query('creative_id==@cid_list'),how='left')
train_full_info=pd.merge(train_full_info,train_usr_label,how='left')
cid_list=test_click_log['creative_id'].to_list()
test_full_info=pd.merge(test_click_log,test_ad_info.query('creative_id==@cid_list'),how='left')


# In[28]:


print("making train...")
train_cal_info=aggregate_features(train_full_info,'agg_')
print("finish train")
train_cal_info.to_csv("../../var/fjw/train_mid/usr_click_log_df_cal.csv",index=False)
print("making test...")
test_cal_info=aggregate_features(test_full_info,'agg_')
print("finsh test...")
test_cal_info.to_csv("../../var/fjw/test_mid/usr_click_log_df_cal.csv",index=False)


# In[46]:


train_cal_info=pd.read_csv("../../var/fjw/train_mid/usr_click_log_df_cal.csv")
test_cal_info=pd.read_csv("../../var/fjw/test_mid/usr_click_log_df_cal.csv")
full_cal_info=pd.concat([train_cal_info,test_cal_info])
full_cal_info.loc[:,full_cal_info.columns[1:]]=full_cal_info[full_cal_info.columns[1:]].apply(lambda x:(x-x.mean())/x.std(),axis=0)
train_cal_info=full_cal_info.iloc[:3000000//test_num].copy()
test_cal_info=full_cal_info.iloc[3000000//test_num:].copy()
train_cal_info.fillna(0,inplace=True)
test_cal_info.fillna(0,inplace=True)
train_cal_info.to_csv("../../var/fjw/train_mid/usr_click_log_df_cal_norm.csv",index=False)
test_cal_info.to_csv("../../var/fjw/test_mid/usr_click_log_df_cal_norm.csv",index=False)


# In[47]:


train_cal_info=pd.read_csv("../../var/fjw/train_mid/usr_click_log_df_cal_norm.csv")
test_cal_info=pd.read_csv("../../var/fjw/test_mid/usr_click_log_df_cal_norm.csv")


# In[48]:


cal_info=pd.concat([train_cal_info,test_cal_info])


# In[50]:


new_cal_info=pd.Series([e for e in cal_info.values[:,1:].astype(np.float32)],index=cal_info['user_id'])


# In[51]:


pk.dump(new_cal_info,open("../../var/fjw/simple_cal_norm.csv","wb"))


# ### TextRank

# In[14]:


import jieba
import jieba.analyse
import jieba.posseg as pseg


# In[15]:


full_info=pd.concat([new_train_usr_click_log.iloc[:,:-2],new_test_usr_click_log])


# In[56]:


tr_cols=['time','creative_id','click_times','ad_id','product_id','product_category','advertiser_id','industry']

for col in tqdm(tr_cols):
    print(col)
    textranks_list= []
    for sequence in tqdm(full_info[col]):
        sentence_list = []
        for item in sequence:
            item = str(item)
            if item == "\\N":
                item = "0000"

            if len(item) == 1:
                sentence_list.append('0' + item)
            else:
                sentence_list.append(item)

        sentence = ' '.join(sentence_list)
    #     print(sentence)

        text_rank_dict = {}
        for x, w in jieba.analyse.textrank(sentence, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v', 'm', 'x'), topK=999999999999):
            text_rank_dict[x] = w 
    #     print(text_rank_dict)

        textranks = []
        for item in sentence_list:
            textranks.append(text_rank_dict[item])
        textranks_list.append(textranks)
    #     break
    pk.dump(textranks_list, open('../../var/fjw/textrank/%s.pickle' % col, 'wb'))


# In[26]:


full_info=pd.concat([new_train_usr_click_log.iloc[:,:-2],new_test_usr_click_log])
creative_id_textrank=pk.load(open("../../var/fjw/textrank/creative_id.pickle","rb"))
ad_id_textrank=pk.load(open("../../var/fjw/textrank/ad_id.pickle","rb"))
product_id_textrank=pk.load(open("../../var/fjw/textrank/product_id.pickle","rb"))
product_category_textrank=pk.load(open("../../var/fjw/textrank/product_category.pickle","rb"))
advertiser_id_textrank=pk.load(open("../../var/fjw/textrank/advertiser_id.pickle","rb"))
industry_textrank=pk.load(open("../../var/fjw/textrank/industry.pickle","rb"))

textrank_info=[creative_id_textrank,ad_id_textrank,product_id_textrank,product_category_textrank,               advertiser_id_textrank,industry_textrank]
full_textrank=[]
indexs=[]
for idx in tqdm(range(len(creative_id_textrank))):
    sample=[]
    for i in range(6):
        sample.append(np.array(textrank_info[i][idx]).astype(np.float32))
    sample=np.stack(sample,axis=1)
    full_textrank.append(sample)
    indexs.append(full_info['user_id'].iloc[idx][0])
full_textrank=pd.Series(data=full_textrank,index=indexs)
pk.dump(full_textrank,open("../../var/fjw/textrank/full_textrank.pickle","wb"))


# ### TFIDF(new)

# #### idf

# In[58]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from category_encoders import *
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
import pickle
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm


# In[60]:


my_var_dir="../../var/hyr"
logging.info('load')
mp_id_se = {
    'creative_id' : pickle.load(open('%s/se_user_creative_id.pickle' % my_var_dir, 'rb')),
    'ad_id' : pickle.load(open('%s/se_user_ad_id.pickle' % my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_user_advertiser_id.pickle'% my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_user_industry.pickle'% my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_user_product_category.pickle' % my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_user_product_id.pickle' % my_var_dir, 'rb')),
    'time' : pickle.load(open('%s/se_user_product_id.pickle' % my_var_dir, 'rb')),
    'click_times' : pickle.load(open('%s/se_user_click_times.pickle' % my_var_dir, 'rb')),
}
logging.info('finish load')


# In[62]:



feature_name = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'time', 'click_times']

def generate_idf_sequence(se_data):
    cross_corpus = list(se_data.apply(lambda a_list : ' '.join([str(x) for x in a_list])))
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"\S+", lowercase = False)
    tfidf_spare = tfidf_vectorizer.fit_transform(cross_corpus)
    idf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
    se_idf = pd.Series(idf_dict)
    pickle.dump(se_idf, open('%s/%s_idf.pickle' % (my_var_dir, column_name), 'wb'))

for column_name in feature_name:
    logging.info('start tfidf sequence %s' % column_name)
    se = mp_id_se[column_name]
    generate_idf_sequence(se)
    logging.info('finish tfidf sequence %s' % column_name)


# In[81]:


ad_id_idf_wv_model=pk.load(open("../../var/hyr/ad_id_idf.pickle","rb"))
creative_id_idf_wv_model=pk.load(open("../../var/hyr/creative_id_idf.pickle","rb"))
product_id_idf_wv_model=pk.load(open("../../var/hyr/product_id_idf.pickle","rb"))
product_category_idf_wv_model=pk.load(open("../../var/hyr/product_category_idf.pickle","rb"))
advertiser_idf_wv_model=pk.load(open("../../var/hyr/advertiser_id_idf.pickle","rb"))
industry_idf_wv_model=pk.load(open("../../var/hyr/industry_idf.pickle","rb"))
if '\\N' in industry_idf_wv_model.index:
    industry_idf_wv_model['0']=industry_idf_wv_model['\\N']
idf_models=[creative_id_idf_wv_model,ad_id_idf_wv_model,product_id_idf_wv_model,product_category_idf_wv_model,           advertiser_idf_wv_model,industry_idf_wv_model]


# In[82]:


train_ad_info=pd.read_csv("../../var/data/train_preliminary/ad.csv")
test_ad_info=pd.read_csv("../../var/data/test/ad.csv")
new_train_ad_info=pd.read_csv("../../var/data/train_semi_final/ad.csv")
train_ad_info=pd.concat([train_ad_info,new_train_ad_info]).drop_duplicates()
full_info=pd.concat([train_ad_info,test_ad_info])
full_info=full_info.drop_duplicates()
full_info.to_csv("../../var/fjw/full_ad.csv",index=False)


# In[83]:


full_info=pd.read_csv("../../var/fjw/full_ad.csv")
target_map_dict=dict()
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    idxs=[ '0' if e=='\\N' else str(e) for e in sample.tolist()]
    value=[]
    for i in range(6):
        value.append(idf_models[i][idxs[i]])
    value=np.array(value)
    target_map_dict[idxs[0]]=value
keys=[]
values=[]
for key,value in tqdm(target_map_dict.items()):
    keys.append(key)
    values.append(value)
info=pd.Series(data=values,index=keys)
pk.dump(info,open("../../var/fjw/full_idf.pickle","wb"))
idf_info=info


# #### tf

# In[86]:


full_info=pd.concat([new_train_usr_click_log.iloc[:,:-2],new_test_usr_click_log])
categorical_cols=['creative_id', 'ad_id','product_id','product_category','advertiser_id','industry']
tf_lists=[]
def find_info(usr_id,sample):
    info=[]
    for i,col in enumerate(categorical_cols):
        num_dict=defaultdict(int)
        
        for e in sample[col]:
            num_dict[str(e)]+=1
        tfs=[]
        for e in sample[col]:
            tfs.append(num_dict[str(e)])
        info.append(tfs)
    return usr_id,np.array(info).transpose()
def push(info):
    usr_id,id_info=info
    if usr_id%100000==0:
        print(usr_id)
    tf_lists.append([usr_id,id_info])
pool=multiprocessing.Pool(8)
indexs=[e[0] for e in full_info['user_id'].values]
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    usr_id=indexs[idx]
#     push(find_info(usr_id,sample))
    pool.apply_async(find_info,args=[usr_id,sample],callback=push)
pool.close()
pool.join()
new_tf_lists=list(sorted(tf_lists,key=lambda x:x[0]))
new_tf_lists=[e[1] for e in new_tf_lists]
tf_info=pd.Series(data=new_tf_lists,index=indexs)
pk.dump(tf_info,open("../../var/fjw/full_tf.pickle","wb"))


# #### 合成

# In[84]:


full_click_log_creative_id=pk.load(open("../../var/fjw/usr_seq/se_user_creative_id.pickle","rb"))


# In[87]:


tfidf_lists=[]
indexs=tf_info.index
for usr_id in tqdm(tf_info.index):
    tf_sample=tf_info[usr_id]
    idf_sample=np.stack(idf_info[[str(e) for e in full_click_log_creative_id.loc[usr_id]]].tolist())
    tf_idf_sample=(tf_sample*idf_sample).astype(np.float32)
    tfidf_lists.append(tf_idf_sample)
tfidf_info=pd.Series(data=tfidf_lists,index=indexs)

pk.dump(tfidf_info,open("../../var/fjw/full_tfidf.pickle","wb"))


# ## feature make

# ### 合成target_info

# In[2]:


usr_id_dict,target_map_dicts=pk.load(open("../../var/fjw/train_mid/target_info.pk","rb"))


# In[3]:


full_info=pd.read_csv("../../var/fjw/full_ad.csv")


# In[4]:



usrs=[[],[],[],[],[],[]]
for key,value in usr_id_dict.items():
    usrs[value].append(key)
cat_cols=['creative_id',
 'ad_id',
 'product_id',
 'product_category',
 'advertiser_id',
 'industry']
new_dicts=[dict() for i in range(6)]
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    idxs=[ '0' if e=='\\N' else str(e) for e in sample.tolist()]
    for fold in range(6):
        flag=True
        for i,idx in enumerate(idxs):
            if idx not in target_map_dicts[fold][cat_cols[i]].index:
                    flag=False
                    break
        if flag:
            values=[]
            for i,idx in enumerate(idxs):
                values.append(target_map_dicts[fold][cat_cols[i]].at[idx])
            values=np.concatenate(values)
            new_dicts[fold]["_".join(idxs)]=values

target_map_dicts=new_dicts
new_dicts=[]
for idx in range(len(target_map_dicts)):
    keys=[]
    print(idx)
    values=[]
    for key,value in tqdm(target_map_dicts[idx].items()):
        keys.append(key)
        values.append(value)
    info=pd.Series(data=values,index=keys)
    new_dicts.append(info)
pk.dump([usr_id_dict,new_dicts],open("../../var/fjw/train_mid/new_target_info.pk","wb"))


# In[5]:


old_usr_dict,old_target_map_dicts=pk.load(open("../../var/fjw/train_mid/target_info.pk","rb"))
usr_dict,target_map_dicts=pk.load(open("../../var/fjw/train_mid/new_target_info.pk","rb"))
for idx in range(len(target_map_dicts)):
    new_index=[e.split("_")[0] for e in list(target_map_dicts[idx].index)]
    target_map_dicts[idx].index=new_index
# pk.dump([usr_dict,target_map_dicts],open("../../var/fjw/train_mid/new_target_info_simple.pk","wb"))
# old_usr_dict,old_target_map_dicts=pk.load(open("../../var/fjw/train_mid/target_info.pk","rb"))
# usr_dict,target_map_dicts=pk.load(open("../../var/fjw/train_mid/new_target_info_simple.pk","rb"))
for idx in tqdm(range(len(old_target_map_dicts))):
    for key in old_target_map_dicts[idx].keys():
        print(key)
        indexs=old_target_map_dicts[idx][key].index.tolist()
        for index in tqdm(indexs):
            old_target_map_dicts[idx][key].loc[index]=old_target_map_dicts[idx][key].loc[index].astype(np.float32)
for idx in tqdm(range(len(target_map_dicts))):
    indexs=target_map_dicts[idx].index.tolist()
    for index in tqdm(indexs):
        target_map_dicts[idx].loc[index]=target_map_dicts[idx].loc[index].astype(np.float32)
pk.dump([old_usr_dict,old_target_map_dicts],open("../../var/fjw/train_mid/target_info.pk","wb"))
pk.dump([usr_dict,target_map_dicts],open("../../var/fjw/train_mid/new_target_info_simple.pk","wb"))


# In[6]:


for i in range(6):
    target_map_dicts[i].index = list(map(lambda x:int(x), target_map_dicts[i].index))
pk.dump([usr_dict,target_map_dicts],open("../../var/hyr/target_info_new.pickle","wb"))


# ### 合成feature_simple

# In[98]:


indexs=[e[0] for e in new_train_usr_click_log['user_id'].values]
ages=[e[0] for e in new_train_usr_click_log['age'].values]
genders=[e[0] for e in new_train_usr_click_log['gender'].values]
new_train_features=[]
for idx in tqdm(range(len(indexs))):
    new_train_features.append([ages[idx],genders[idx]])
new_train_features=pd.Series(data=new_train_features,index=indexs)   
pk.dump(new_train_features,open("../../var/fjw/train_mid/new_feature_simple.pk","wb"))


# In[100]:



indexs=[e[0] for e in new_test_usr_click_log['user_id'].values]
new_test_features=[]
for idx in tqdm(range(len(indexs))):
    new_test_features.append([1,1])
new_test_features=pd.Series(data=new_test_features,index=indexs)   
pk.dump(new_test_features,open("../../var/fjw/test_mid/new_feature_simple.pk","wb"))


# ### 训练glove50维及200维(请前往./Glove_Master/目录下执行Generate_glove.ipynb)

# ### 合成word2vec(请完成./src/hyr/下的Word2vec 训练以及4.3步的Glove训练后前往w2v_merge.ipynb执行
