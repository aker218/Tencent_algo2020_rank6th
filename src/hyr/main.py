#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import datetime
import pickle
import scipy.sparse as ss
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import seaborn as sns

import IPython.display as ipd
import copy
import random
# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize(progress_bar=True)
# df.parallel_apply(func)
from tqdm import tqdm_notebook as tqdm

from gensim.models.word2vec import Word2Vec 
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold


# # Read

# In[2]:


var_dir = '../../var'
my_var_dir = '../../var/hyr'
fjw_var_dir = '../../var/fjw'


# In[3]:


df_train_click = pd.read_csv('%s/data/train_semi_final/click_log.csv' % var_dir)
df_train_ad = pd.read_csv('%s/data/train_semi_final/ad.csv' % var_dir)

df_test_click = pd.read_csv('%s/data/test/click_log.csv' % var_dir)
df_test_ad = pd.read_csv('%s/data/test/ad.csv' % var_dir)


# In[4]:


pre_df_train_click = pd.read_csv('%s/data/train_preliminary/click_log.csv' % var_dir)
pre_df_train_ad = pd.read_csv('%s/data/train_preliminary/ad.csv' % var_dir)
df_train_click = pd.concat([pre_df_train_click, df_train_click])
df_train_ad = pd.concat([pre_df_train_ad, df_train_ad])


# In[5]:


df_train_user = pd.read_csv('%s/data/train_semi_final/user.csv'% var_dir)
pre_df_train_user = pd.read_csv('%s/data/train_preliminary/user.csv'% var_dir)
df_train_user = pd.concat([pre_df_train_user, df_train_user])


# # preprocess

# ## dump

# ### 存整个

# In[6]:


df_train_click['type'] = 'train'
df_test_click['type'] = 'test'
df_click = pd.concat([df_train_click, df_test_click])
df_ad = pd.concat([df_train_ad, df_test_ad])
df_ad = df_ad.drop_duplicates()
df_click = df_click.sort_values(by = ['time'])


# In[7]:


gby_user_id = df_click[['user_id', 'creative_id']].groupby('user_id')

#se_user_creative_id
mp_user_creative = {}
for user, df_item in tqdm(gby_user_id):
    mp_user_creative[user] = list(df_item['creative_id'])
se_user_creative_id = pd.Series(mp_user_creative)
pickle.dump(se_user_creative_id, open('%s/se_user_creative_id.pickle' % my_var_dir, 'wb'))

#se_user_time, se_user_click_times
gby_user_id = df_click[['user_id', 'time', 'click_times']].groupby('user_id')
mp_user_time = {}
mp_user_click_times = {}
for user, df_item in tqdm(gby_user_id):
    mp_user_time[user] = list(df_item['time'])
    mp_user_click_times[user] = list(df_item['click_times'])
se_user_time = pd.Series(mp_user_time)
se_user_click_times = pd.Series(mp_user_click_times)
pickle.dump(se_user_time, open('%s/se_user_time.pickle' % my_var_dir, 'wb'))
pickle.dump(se_user_click_times, open('%s/se_user_click_times.pickle' % my_var_dir, 'wb'))


df_ad.index = df_ad.creative_id
for feature in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
    pickle.dump(df_ad[feature], open('%s/se_creative_id_%s.pickle' % ( my_var_dir, feature), 'wb'))


# In[8]:


logging.info('load')

se_user_creative_id = pickle.load(open('%s/se_user_creative_id.pickle' % my_var_dir, 'rb'))
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'% my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'% my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_creative_id_industry.pickle'% my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_creative_id_product_category.pickle'% my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_creative_id_product_id.pickle'% my_var_dir, 'rb')),
}
logging.info('finish load')


se_user_creative_id_origin = None
def sub_creative_id():
    global se_user_creative_id, se_user_creative_id_origin
    assert se_user_creative_id.shape[0] == 4000000
    se_user_creative_id_origin = se_user_creative_id
    se_user_creative_id = se_user_creative_id.iloc[:1024 * 5]
    
def full_creative_id():
    global se_user_creative_id, se_user_creative_id_origin
    se_user_creative_id = se_user_creative_id_origin
    
def get_feature_se(name):
    if(name == "creative_id"):
        return se_user_creative_id
    
    mp_user_other = {}
    for user in tqdm(se_user_creative_id.index):
        mp_user_other[user] = list(mp_creative_other[name][se_user_creative_id[user]])
    se_user_other = pd.Series(mp_user_other)
    return se_user_other


# In[9]:


mp_creative_other['product_id'] = mp_creative_other['product_id'].replace("\\N", 0)
mp_creative_other['industry'] = mp_creative_other['industry'].replace("\\N", 0)
mp_creative_other['product_id'] = mp_creative_other['product_id'].apply(lambda x : int(x))
mp_creative_other['industry'] = mp_creative_other['industry'].apply(lambda x : int(x))


# In[10]:


for feature in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
    se = get_feature_se(feature)
    pickle.dump(se, open('%s/se_user_%s.pickle' % (my_var_dir, feature), 'wb'))


# ## load

# In[11]:


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


# # merge df_user_info

# In[12]:


import pickle
se_user_creative_id = pickle.load(open('%s/se_user_creative_id.pickle'%my_var_dir, 'rb'))
se_user_time = pickle.load(open('%s/se_user_time.pickle'%my_var_dir, 'rb'))
se_user_click_times = pickle.load(open('%s/se_user_click_times.pickle'%my_var_dir, 'rb'))
new_seq_list = []
for seq in se_user_click_times:
    new_seq = list(map(lambda x : x if x < 32 else 32, seq))
    new_seq_list.append(new_seq)
se_user_click_times_cut = pd.Series(new_seq_list)
se_user_click_times_cut.index = se_user_click_times.index
df_user_info = pd.DataFrame(list(zip(se_user_creative_id, se_user_time, se_user_click_times_cut)), columns = ['creative_id', 'time', 'click_time'])
df_user_info.index = se_user_creative_id.index
pickle.dump(df_user_info, open('%s/df_user_info.pickle' % my_var_dir, 'wb'))


# # tfidf

# In[13]:


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


# ## sequence (idf map)

# In[14]:


feature_name = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'time', 'click_times']

def generate_idf_sequence(se_data):
    cross_corpus = list(se_data.apply(lambda a_list : ' '.join([str(x) for x in a_list])))
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"\S+", lowercase = False)
    tfidf_spare = tfidf_vectorizer.fit_transform(cross_corpus)
    idf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf_vectorizer.idf_))
    se_idf = pd.Series(idf_dict)
    ipd.display(se_idf.iloc[:10])
    pickle.dump(se_idf, open('%s/%s_idf.pickle' % (my_var_dir, column_name), 'wb'))

for column_name in feature_name:
    logging.info('start tfidf sequence %s' % column_name)
    se = mp_id_se[column_name]
    generate_idf_sequence(se)
    logging.info('finish tfidf sequence %s' % column_name)


# ## stack feature

# In[15]:


def generate_tfidf(column_name):
    se = mp_id_se[column_name]
    cross_corpus = list(se.apply(lambda a_list : ' '.join(['%s_'%column_name + str(x) for x in a_list])))
    tfidf_vectorizer = TfidfVectorizer(max_features = 50000)
    tfidf_spare = tfidf_vectorizer.fit_transform(cross_corpus)
    return tfidf_spare

def generate_count_word(column_name):
    se = mp_id_se[column_name]
    cross_corpus = list(se.apply(lambda a_list : ' '.join(['%s_'%column_name + str(x) for x in a_list])))
    count_vectorizer = CountVectorizer(max_features = 50000)
    count_spare = count_vectorizer.fit_transform(cross_corpus)
    return count_spare

feature_name = ['time', 'creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
spare_m_list = []
logging.info('start cat_matrix')
for name in feature_name:
    tfidf = generate_tfidf(name)
#     count_word = generate_count_word(name)
    spare_m_list.append(tfidf)
#     spare_m_list.append(count_word)  
    logging.info('finish %s' % name)
    print(tfidf.shape)

logging.info('finish cat_matrix')

cat_matrix = scipy.sparse.hstack(spare_m_list)
cat_matrix.shape


# In[16]:


# df_train_user = pd.read_csv('/data/ccnth/algo/train_semi_final/user.csv')
# pre_df_train_user = pd.read_csv('/data/ccnth/algo/train_preliminary/user.csv')
# df_labels = pd.concat([pre_df_train_user, df_train_user])
df_labels = df_train_user
cat_matrix = cat_matrix.tocsr()


# In[18]:


def tfidf_count_cross_validation(train_x, test_x, y, model_class, n_class):
    
    n_flod = 5
    folds = KFold(n_splits=n_flod, shuffle=True, random_state=0)
    
    if n_class > 2:
        val_score = np.zeros((train_x.shape[0], n_class))
        test_score = np.zeros((test_x.shape[0], n_class))
    else:
        val_score = np.zeros((train_x.shape[0], 1))
        test_score = np.zeros((test_x.shape[0], 1))
        
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x, y)):
        logging.info('fold:%d' % n_fold)
        model = model_class()
        trn_x, trn_y = train_x[trn_idx], y[trn_idx]
        val_x, val_y = train_x[val_idx], y[val_idx]
        model.fit(trn_x, trn_y)
        
        try:
            val_ret = model.predict_proba(val_x)
            test_ret = model.predict_proba(test_x)
            if(n_class == 2):
                val_ret = val_ret[:, 1]
                test_ret = test_ret[:, 1]
        except:
            val_ret = model.decision_function(val_x)
            test_ret = model.decision_function(test_x)

        
        if(n_class == 2):
            val_ret = val_ret.reshape(-1, 1)
            test_ret = test_ret.reshape(-1, 1)

        val_score[val_idx] = val_ret
        test_score += test_ret / n_flod
        
    return val_score, test_score


model_classes = [SGDClassifier, PassiveAggressiveClassifier, BernoulliNB, MultinomialNB, LogisticRegression]
#ok : SGDClassifier, PassiveAggressiveClassifier, BernoulliNB, MultinomialNB, LogisticRegression
#not ok RidgeClassifier, DecisionTreeClassifier RandomForestClassifier ExtraTreesClassifier
# model_classes = [ SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, BernoulliNB, MultinomialNB]

train_score_list, test_score_list = [], []

n = 1
for i, model_class in enumerate(model_classes):    
    logging.info('start: %d' % i)
    val_score, test_score =     tfidf_count_cross_validation(cat_matrix[:int(3000000/n)], cat_matrix[int(3000000/n):], df_labels['gender'].values, model_class, 2)
    train_score_list.append(val_score)
    test_score_list.append(test_score)
    
    val_score, test_score =     tfidf_count_cross_validation(cat_matrix[:int(3000000/n)], cat_matrix[int(3000000/n):], df_labels['age'].values, model_class, 10)
    train_score_list.append(val_score)
    test_score_list.append(test_score)


# In[19]:


train_stack = np.concatenate(train_score_list, axis = 1)
test_stack = np.concatenate(test_score_list, axis = 1)
se_stack = pd.Series(list(np.concatenate([train_stack, test_stack])))
se_stack.index = mp_id_se['ad_id'].index
pickle.dump(se_stack, open('%s/se_tfidf_stack_new.pickle' % my_var_dir, 'wb'))


# In[20]:


pickle.dump(se_stack, open('%s/se_tfidf_stack_new.pickle' % fjw_var_dir, 'wb'))


# In[21]:


# for i in range(int(len(train_score_list) / 2)):
#     gender_acc, age_acc = accuracy_score(train_score_list[2*i][:, 0] > 0.5, df_labels['gender'] - 1), \
#     accuracy_score(np.argmax(train_score_list[2*i+1], axis = 1), df_labels['age'] - 1)
#     logging.info('%d : gender : %f, age : %f' % (i, gender_acc, age_acc))


# # wv

# In[21]:


import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import datetime
import pickle
import scipy.sparse as ss
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
# import seaborn as sns

import IPython.display as ipd
import copy
import random
from gensim.models.word2vec import Word2Vec 
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold


# In[22]:


window = 175
def generate_wv(name):
    se_user_other = pickle.load(open('%s/se_user_%s.pickle' % (my_var_dir, name), 'rb'))
    
    sentences = []
    for a_user in se_user_other:
        sentence = []
        for item_id in a_user:
            sentence.append(str(item_id))
        sentences.append(sentence)

    wv_model= Word2Vec(sentences, min_count=1, size = mp_column_nembedding[name], window = window, sg = 1, iter = 5, workers = 16)
    
    return wv_model


# ## single 200 100 dim

# In[22]:


mp_column_nembedding = {
    'creative_id' : 200, 
    'ad_id' : 200, 
    'product_id' : 100, 
    'product_category' : 100,
    'advertiser_id' : 200,
    'industry' : 100
}
used_col = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
for col in used_col:
    logging.info('start %s wv' % col)
    wv_model = generate_wv(col)
    pickle.dump(wv_model, open('%s/wv_%s_sg_window%d_dim%d.pickle' % (var_dir, col, window, mp_column_nembedding[col]), 'wb'))
    logging.info('finish %s wv' % col)


# In[29]:


mp_column_nembedding = {
    'creative_id' : 100, 
    'ad_id' : 100, 
    'advertiser_id' : 100,
}
used_col = ['creative_id', 'ad_id', 'advertiser_id',]
for col in used_col:
    logging.info('start %s wv' % col)
    wv_model = generate_wv(col)
    pickle.dump(wv_model, open('%s/wv_%s_sg_window%d_dim%d.pickle' % (var_dir, col, window, mp_column_nembedding[col]), 'wb'))
    logging.info('finish %s wv' % col)


# ## single 50 dim

# In[23]:


mp_column_nembedding = {
    'creative_id' : 50, 
    'ad_id' : 50, 
    'product_id' : 50, 
    'product_category' : 50,
    'advertiser_id' : 50,
    'industry' : 50
}
used_col = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
for col in used_col:
    logging.info('start %s wv' % col)
    wv_model = generate_wv(col)
    pickle.dump(wv_model, open('%s/wv_%s_sg_window%d_dim%d.pickle' % (var_dir, col, window, mp_column_nembedding[col]), 'wb'))
    logging.info('finish %s wv' % col)


# ## epoch 16

# In[23]:


window = 175
def generate_wv_epoch16(name):
    se_user_other = pickle.load(open('%s/se_user_%s.pickle' % (my_var_dir, name), 'rb'))
    
    sentences = []
    for a_user in se_user_other:
        sentence = []
        for item_id in a_user:
            sentence.append(str(item_id))
        sentences.append(sentence)

    wv_model= Word2Vec(sentences, min_count=1, size = mp_column_nembedding[name], window = window, sg = 1, iter = 16, workers = 16)
    
    return wv_model

mp_column_nembedding = {
    'ad_id' : 200, 
    'advertiser_id' : 200,
}
used_col = [ 'ad_id',   'advertiser_id']
for col in used_col:
    logging.info('start %s wv' % col)
    wv_model = generate_wv_epoch16(col)
    pickle.dump(wv_model, open('%s/wv_%s_sg_window%d_dim%d_epoch16.pickle' % (var_dir, col, window, mp_column_nembedding[col]), 'wb'))
    logging.info('finish %s wv' % col)


# ## cross

# In[24]:


window = 175
cross_id_set = set()
def word2vec_cross(se_feature_sequence1, se_feature_sequence2, n_dim):
    sentences = []
    for f1, f2 in zip(se_feature_sequence1, se_feature_sequence2):
        sentence = []
        for item_id1, item_id2, in zip(f1, f2):
            cross_id = "%s_%s"%(str(item_id1), str(item_id2))
            sentence.append(cross_id)
        sentences.append(sentence)
        cross_id_set.add(cross_id)
        
    logging.info('unique id:%d' % len(set(cross_id_set)))
    wv_model= Word2Vec(sentences, min_count=1, size = n_dim, window = window, sg = 1, workers = 24)
    

    return wv_model

def generate_cross(feature1, feature2, n_dim = 50):
    se_feature_sequence1 = mp_id_se[feature1]
    se_feature_sequence2 = mp_id_se[feature2]
    logging.info('start corss %s %s wv' % (feature1, feature2))
    model = word2vec_cross(se_feature_sequence1, se_feature_sequence2, n_dim)
    pickle.dump(model, open('%s/cross_wv_model_%s_%s_sg_window%d_dim%d.pickle' %                             (var_dir, feature1, feature2, window, n_dim), 'wb'))
    logging.info('finish corss %s %s wv' % (feature1, feature2))


# In[25]:


generate_cross('product_id', 'advertiser_id')
generate_cross('product_id', 'industry')
generate_cross('product_id', 'product_category')
generate_cross('product_category', 'advertiser_id')
generate_cross('product_category', 'industry')
generate_cross('advertiser_id', 'industry')


# In[26]:


generate_cross('product_id', 'advertiser_id', 100)
generate_cross('product_id', 'industry', 100)
generate_cross('product_id', 'product_category', 100)
generate_cross('product_category', 'advertiser_id', 100)
generate_cross('product_category', 'industry', 100)
generate_cross('advertiser_id', 'industry', 100)


# In[ ]:




