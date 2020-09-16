#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# ## 50维

# In[5]:


creative_id_glove_model=pk.load(open("../../var/se_glove_creative_id_50dim.pickle","rb"))
ad_id_glove_model=pk.load(open("../../var/se_glove_ad_id_50dim.pickle","rb"))
advertiser_id_glove_model=pk.load(open("../../var/se_glove_advertiser_id_50dim.pickle","rb"))
product_id_glove_model=pk.load(open("../../var/se_glove_product_id_50dim.pickle","rb"))
product_category_glove_model=pk.load(open("../../var/se_glove_product_category_50dim.pickle","rb"))
industry_glove_model=pk.load(open("../../var/se_glove_industry_50dim.pickle","rb"))
advertiser_id_industry_glove_model=pk.load(open("../../var/se_glove_advertiser_id_industry_50dim.pickle","rb"))
product_category_advertiser_glove_model=pk.load(open("../../var/se_glove_product_category_advertiser_id_50dim.pickle","rb"))
product_category_industry_glove_model=pk.load(open("../../var/se_glove_product_category_industry_50dim.pickle","rb"))
product_id_advertiser_glove_model=pk.load(open("../../var/se_glove_product_id_advertiser_id_50dim.pickle","rb"))
product_id_industry_glove_model=pk.load(open("../../var/se_glove_product_id_industry_50dim.pickle","rb"))
product_id_product_category_glove_model=pk.load(open("../../var/se_glove_product_id_product_category_50dim.pickle","rb"))


# In[6]:


full_info=pd.read_csv("../../var/fjw/full_ad.csv")

# glove_models=[creative_id_glove_model,ad_id_glove_model,product_id_glove_model,product_category_glove_model,\
#             advertiser_id_glove_model,industry_glove_model,product_id_advertiser_glove_model,\
#             product_category_industry_glove_model]
glove_models=[creative_id_glove_model,ad_id_glove_model,product_id_glove_model,product_category_glove_model,            advertiser_id_glove_model,industry_glove_model,advertiser_id_industry_glove_model,            product_category_advertiser_glove_model,product_category_industry_glove_model,            product_id_advertiser_glove_model,product_id_industry_glove_model,product_id_product_category_glove_model]


# In[7]:


import gc
target_map_dict=dict()
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    idxs=[ '0' if e=='\\N' else str(e) for e in sample.tolist()]
    value=[]
    for i in range(len(glove_models)):
        if i<6:
            value.append(glove_models[i][idxs[i]])
        elif i==6:
            value.append(glove_models[i][idxs[4]+"_"+idxs[5]])
        elif i==7:
            value.append(glove_models[i][idxs[3]+"_"+idxs[4]]) 
        elif i==8:
            value.append(glove_models[i][idxs[3]+"_"+idxs[5]]) 
        elif i==9:
            value.append(glove_models[i][idxs[2]+"_"+idxs[4]]) 
        elif i==10:
            value.append(glove_models[i][idxs[2]+"_"+idxs[5]]) 
        elif i==11:
            value.append(glove_models[i][idxs[2]+"_"+idxs[3]]) 
    value=np.concatenate(value)
    target_map_dict["_".join(idxs)]=value.astype(np.float32)
info=pd.Series(target_map_dict)
del target_map_dict,glove_models
gc.collect()


# In[8]:


new_glove_info=np.zeros((4445720+1,600)).astype(np.float32)
for index in tqdm(info.index):
    new_glove_info[int(index.split("_")[0])]=info.loc[index]
np.save("../../var/fjw/glove_model_min_dim50_full.npy",new_glove_info)


# In[9]:


np.save("../../var/hyr/id_embedding_glove50.npy",new_glove_info)


# ## 200维

# In[10]:


creative_id_w2v_model=pk.load(open("../../var/wv_creative_id_sg_window175_dim200.pickle","rb"))
ad_id_w2v_model=pk.load(open("../../var/wv_ad_id_sg_window175_dim200.pickle","rb"))
advertiser_id_w2v_model=pk.load(open("../../var/wv_advertiser_id_sg_window175_dim200.pickle","rb"))
product_id_w2v_model=pk.load(open("../../var/wv_product_id_sg_window175_dim100.pickle","rb"))
product_category_w2v_model=pk.load(open("../../var/wv_product_category_sg_window175_dim100.pickle","rb"))
industry_w2v_model=pk.load(open("../../var/wv_industry_sg_window175_dim100.pickle","rb"))
# advertiser_id_industry_w2v_model=pk.load(open("./wv_dir/wv_model/cross_wv_model_advertiser_id_industry_sg_window175_dim100.pickle","rb"))
# product_category_advertiser_w2v_model=pk.load(open("./wv_dir/wv_model/cross_wv_model_product_category_advertiser_id_sg_window175_dim100.pickle","rb"))
product_category_industry_w2v_model=pk.load(open("../../var/cross_wv_model_product_category_industry_sg_window175_dim100.pickle","rb"))
product_id_advertiser_w2v_model=pk.load(open("../../var/cross_wv_model_product_id_advertiser_id_sg_window175_dim100.pickle","rb"))
# product_id_industry_w2v_model=pk.load(open("./wv_dir/wv_model/cross_wv_model_product_id_industry_sg_window175_dim100.pickle","rb"))
# product_id_product_category_w2v_model=pk.load(open("./wv_dir/wv_model/cross_wv_model_product_id_product_category_sg_window175_dim100.pickle","rb"))


# In[11]:


full_info=pd.read_csv("../../var/fjw/full_ad.csv")
w2v_models=[creative_id_w2v_model,ad_id_w2v_model,product_id_w2v_model,product_category_w2v_model,            advertiser_id_w2v_model,industry_w2v_model,product_id_advertiser_w2v_model,            product_category_industry_w2v_model]
# w2v_models=[creative_id_w2v_model,ad_id_w2v_model,product_id_w2v_model,product_category_w2v_model,\
#             advertiser_id_w2v_model,industry_w2v_model,advertiser_id_industry_w2v_model,\
#             product_category_advertiser_w2v_model,product_category_industry_w2v_model,\
#             product_id_advertiser_w2v_model,product_id_industry_w2v_model,product_id_product_category_w2v_model]


# In[12]:


target_map_dict=dict()
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    idxs=[ '0' if e=='\\N' else str(e) for e in sample.tolist()]
    value=[]
    for i in range(8):
        if i<6:
            value.append(w2v_models[i].wv[idxs[i]])
        elif i==6:
            value.append(w2v_models[i].wv[idxs[2]+"_"+idxs[4]])
        else:
            value.append(w2v_models[i].wv[idxs[3]+"_"+idxs[5]]) 
#     for i in range(8):
#         if i<6:
#             value.append(glove_models[i][idxs[i]])
#         elif i==6:
#             value.append(glove_models[i][idxs[2]+"_"+idxs[4]])
#         else:
#             value.append(glove_models[i][idxs[3]+"_"+idxs[5]]) 
    value=np.concatenate(value)
    target_map_dict["_".join(idxs)]=value.astype(np.float32)
info=pd.Series(target_map_dict)
del target_map_dict,w2v_models
gc.collect()
new_w2v_info=np.zeros((4445720+1,1100)).astype(np.float32)
for index in tqdm(info.index):
    new_w2v_info[int(index.split("_")[0])]=info.loc[index]
np.save("../../var/fjw/w2v_model_dim200.npy",new_w2v_info)


# In[13]:


creative_id_glove_model=pk.load(open("../../var/se_glove_creative_id_200dim.pickle","rb"))
ad_id_glove_model=pk.load(open("../../var/se_glove_ad_id_200dim.pickle","rb"))
advertiser_id_glove_model=pk.load(open("../../var/se_glove_advertiser_id_200dim.pickle","rb"))
product_id_glove_model=pk.load(open("../../var/se_glove_product_id_100dim.pickle","rb"))
product_category_glove_model=pk.load(open("../../var/se_glove_product_category_100dim.pickle","rb"))
industry_glove_model=pk.load(open("../../var/se_glove_industry_100dim.pickle","rb"))
# advertiser_id_industry_glove_model=pk.load(open("./wv_dir/glove_wv/se_glove_advertiser_id_industry.pickle","rb"))
# product_category_advertiser_glove_model=pk.load(open("./wv_dir/glove_wv/se_glove_product_category_advertiser_id.pickle","rb"))
product_category_industry_glove_model=pk.load(open("../../var/se_glove_product_category_industry_100dim.pickle","rb"))
product_id_advertiser_glove_model=pk.load(open("../../var/se_glove_product_id_advertiser_id_100dim.pickle","rb"))
# product_id_industry_glove_model=pk.load(open("./wv_dir/glove_wv/se_glove_product_id_industry.pickle","rb"))
# product_id_product_category_glove_model=pk.load(open("./wv_dir/glove_wv/se_glove_product_id_product_category.pickle","rb"))


# In[14]:


full_info=pd.read_csv("../../var/fjw/full_ad.csv")
glove_models=[creative_id_glove_model,ad_id_glove_model,product_id_glove_model,product_category_glove_model,            advertiser_id_glove_model,industry_glove_model,product_id_advertiser_glove_model,            product_category_industry_glove_model]
# glove_models=[creative_id_glove_model,ad_id_glove_model,product_id_glove_model,product_category_glove_model,\
#             advertiser_id_glove_model,industry_glove_model,advertiser_id_industry_glove_model,\
#             product_category_advertiser_glove_model,product_category_industry_glove_model,\
#             product_id_advertiser_glove_model,product_id_industry_glove_model,product_id_product_category_glove_model]


# In[15]:


target_map_dict=dict()
for idx in tqdm(range(full_info.shape[0])):
    sample=full_info.iloc[idx]
    idxs=[ '0' if e=='\\N' else str(e) for e in sample.tolist()]
    value=[]
#     for i in range(8):
#         if i<6:
#             value.append(w2v_models[i].wv[idxs[i]])
#         elif i==6:
#             value.append(w2v_models[i].wv[idxs[2]+"_"+idxs[4]])
#         else:
#             value.append(w2v_models[i].wv[idxs[3]+"_"+idxs[5]]) 
    for i in range(8):
        if i<6:
            value.append(glove_models[i][idxs[i]])
        elif i==6:
            value.append(glove_models[i][idxs[2]+"_"+idxs[4]])
        else:
            value.append(glove_models[i][idxs[3]+"_"+idxs[5]]) 
    value=np.concatenate(value)
    target_map_dict["_".join(idxs)]=value.astype(np.float32)
info=pd.Series(target_map_dict)


# In[16]:


new_glove_info=np.zeros((4445720+1,1100)).astype(np.float32)
for index in tqdm(info.index):
    new_glove_info[int(index.split("_")[0])]=info.loc[index]
np.save("../../var/fjw/glove_model_200dim.npy",new_glove_info)


# In[17]:


new_glove_info=np.zeros((4445720+1,1100)).astype(np.float32)
for index in tqdm(info.index):
    new_glove_info[int(index.split("_")[0])]=info.loc[index]
np.save("../../var/hyr/glove_model_200dim.npy",new_glove_info)

