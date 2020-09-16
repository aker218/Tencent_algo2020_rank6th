#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from pandarallel import pandarallel
# Initialization
pandarallel.initialize(progress_bar=True)
# df.parallel_apply(func)
from gensim.models.word2vec import Word2Vec 
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold


# In[2]:


var_dir = '../../var'
my_var_dir = '../../var/hyr'


# In[3]:


df_train_user = pd.read_csv('%s/data/train_semi_final/user.csv'% var_dir)
pre_df_train_user = pd.read_csv('%s/data/train_preliminary/user.csv'% var_dir)
df_train_user = pd.concat([pre_df_train_user, df_train_user])
offline = df_train_user.shape[0] < 30000


# # wv id_embedding_caa_200

# In[4]:


logging.info('load wv')
wv_model_dict = {
    'creative_id' : pickle.load(open('%s/wv_creative_id_sg_window175_dim200.pickle' % var_dir, 'rb')),
    'ad_id' : pickle.load(open('%s/wv_ad_id_sg_window175_dim200.pickle' % var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/wv_product_id_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_category' : pickle.load(open('%s/wv_product_category_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'advertiser_id' :  pickle.load(open('%s/wv_advertiser_id_sg_window175_dim200.pickle'% var_dir, 'rb')), 
    'industry' : pickle.load(open('%s/wv_industry_sg_window175_dim100.pickle'% var_dir, 'rb')),
    
    'advertiser_id_industry' : pickle.load(open('%s/cross_wv_model_advertiser_id_industry_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_category_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_category_advertiser_id_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_category_industry' : pickle.load(open('%s/cross_wv_model_product_category_industry_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_id_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_id_advertiser_id_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_id_industry' : pickle.load(open('%s/cross_wv_model_product_id_industry_sg_window175_dim100.pickle'% var_dir, 'rb')), 
    'product_id_product_category' : pickle.load(open('%s/cross_wv_model_product_id_product_category_sg_window175_dim100.pickle'% var_dir, 'rb')), 
}
logging.info('finish wv')

logging.info('load')
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'% my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'% my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_creative_id_industry.pickle'% my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_creative_id_product_category.pickle'% my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_creative_id_product_id.pickle'% my_var_dir, 'rb')),
}
logging.info('finish load')

mp_creative_other['product_id'] = mp_creative_other['product_id'].replace("\\N", 0)
mp_creative_other['industry'] = mp_creative_other['industry'].replace("\\N", 0)
mp_creative_other['product_id'] = mp_creative_other['product_id'].apply(lambda x : int(x))
mp_creative_other['industry'] = mp_creative_other['industry'].apply(lambda x : int(x))


# In[5]:


embedding = np.zeros((4445721, 1000)).astype('float32')
if offline == False:
    creative_ids_str = [str(i) for i in range(1, 4445721)]
    creative_ids = list(range(1, 4445721))

    product_id_str = [ str(item) for item in mp_creative_other['product_id'][creative_ids].values]
    product_category_str = [ str(item) for item in mp_creative_other['product_category'][creative_ids].values]
    industry_str = [ str(item) for item in mp_creative_other['industry'][creative_ids].values]


    embedding[1:, : 200] = wv_model_dict['creative_id'].wv[creative_ids_str]

    ad_id_str = [ str(item) for item in mp_creative_other['ad_id'][creative_ids].values]
    embedding[1:, 200 : 400] = wv_model_dict['ad_id'].wv[ad_id_str]

    advertiser_id_str = [ str(item) for item in mp_creative_other['advertiser_id'][creative_ids].values]
    embedding[1:, 400 : 600] = wv_model_dict['advertiser_id'].wv[advertiser_id_str]

    advertiser_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(advertiser_id_str, industry_str)]
    embedding[1:, 600 : 700] = wv_model_dict['advertiser_id_industry'].wv[advertiser_id_industry_str]

    product_category_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, advertiser_id_str)]
    embedding[1:, 700 : 800] = wv_model_dict['product_category_advertiser_id'].wv[product_category_advertiser_id_str]


    product_id_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, advertiser_id_str)]
    embedding[1:, 800 : 900] = wv_model_dict['product_id_advertiser_id'].wv[product_id_advertiser_id_str]


    product_id_product_category_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, product_category_str)]
    embedding[1:, 900 : 1000] = wv_model_dict['product_id_product_category'].wv[product_id_product_category_str]

logging.info('start save id embedding')
np.save('%s/id_embedding_caa_200.npy' % my_var_dir, embedding)
logging.info('finish save id embedding')


# # wv id_embedding50

# In[20]:


logging.info('load wv')
wv_model_dict = {
    'creative_id' : pickle.load(open('%s/wv_creative_id_sg_window175_dim50.pickle' % var_dir, 'rb')),
    'ad_id' : pickle.load(open('%s/wv_ad_id_sg_window175_dim50.pickle' % var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/wv_product_id_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_category' : pickle.load(open('%s/wv_product_category_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'advertiser_id' :  pickle.load(open('%s/wv_advertiser_id_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'industry' : pickle.load(open('%s/wv_industry_sg_window175_dim50.pickle'% var_dir, 'rb')),
    
    'advertiser_id_industry' : pickle.load(open('%s/cross_wv_model_advertiser_id_industry_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_category_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_category_advertiser_id_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_category_industry' : pickle.load(open('%s/cross_wv_model_product_category_industry_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_id_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_id_advertiser_id_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_id_industry' : pickle.load(open('%s/cross_wv_model_product_id_industry_sg_window175_dim50.pickle'% var_dir, 'rb')), 
    'product_id_product_category' : pickle.load(open('%s/cross_wv_model_product_id_product_category_sg_window175_dim50.pickle'% var_dir, 'rb')), 
}
logging.info('finish wv')

logging.info('load')
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'% my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'% my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_creative_id_industry.pickle'% my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_creative_id_product_category.pickle'% my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_creative_id_product_id.pickle'% my_var_dir, 'rb')),
}
logging.info('finish load')

mp_creative_other['product_id'] = mp_creative_other['product_id'].replace("\\N", 0)
mp_creative_other['industry'] = mp_creative_other['industry'].replace("\\N", 0)
mp_creative_other['product_id'] = mp_creative_other['product_id'].apply(lambda x : int(x))
mp_creative_other['industry'] = mp_creative_other['industry'].apply(lambda x : int(x))


# In[ ]:


embedding = np.zeros((4445721, 600)).astype('float32')
if offline == False:
    creative_ids_str = [str(i) for i in range(1, 4445721)]
    creative_ids = list(range(1, 4445721))

    embedding[1:, : 50] = wv_model_dict['creative_id'].wv[creative_ids_str]

    ad_id_str = [ str(item) for item in mp_creative_other['ad_id'][creative_ids].values]
    embedding[1:, 50 : 100] = wv_model_dict['ad_id'].wv[ad_id_str]

    product_id_str = [ str(item) for item in mp_creative_other['product_id'][creative_ids].values]
    embedding[1:, 100 : 150] = wv_model_dict['product_id'].wv[product_id_str]

    product_category_str = [ str(item) for item in mp_creative_other['product_category'][creative_ids].values]
    embedding[1:, 150 : 200] = wv_model_dict['product_category'].wv[product_category_str]

    advertiser_id_str = [ str(item) for item in mp_creative_other['advertiser_id'][creative_ids].values]
    embedding[1:, 200 : 250] = wv_model_dict['advertiser_id'].wv[advertiser_id_str]

    industry_str = [ str(item) for item in mp_creative_other['industry'][creative_ids].values]
    embedding[1:, 250 : 300] = wv_model_dict['industry'].wv[industry_str]


    advertiser_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(advertiser_id_str, industry_str)]
    embedding[1:, 300 : 350] = wv_model_dict['advertiser_id_industry'].wv[advertiser_id_industry_str]

    product_category_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, advertiser_id_str)]
    embedding[1:, 350 : 400] = wv_model_dict['product_category_advertiser_id'].wv[product_category_advertiser_id_str]

    product_category_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, industry_str)]
    embedding[1:, 400 : 450] = wv_model_dict['product_category_industry'].wv[product_category_industry_str]

    product_id_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, advertiser_id_str)]
    embedding[1:, 450 : 500] = wv_model_dict['product_id_advertiser_id'].wv[product_id_advertiser_id_str]

    product_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, industry_str)]
    embedding[1:, 500 : 550] = wv_model_dict['product_id_industry'].wv[product_id_industry_str]

    product_id_product_category_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, product_category_str)]
    embedding[1:, 550 : 600] = wv_model_dict['product_id_product_category'].wv[product_id_product_category_str]

logging.info('start save id embedding')
np.save('%s/id_embedding50.npy'%my_var_dir, embedding)
logging.info('finish save id embedding')


# # id_embedding_glove50、id_embedding_glove_short.npy、id_embedding_glove_short_pair.npy

# In[24]:


id_embeddings = np.load('%s/id_embedding_glove50.npy'% my_var_dir)
creative_id = id_embeddings[:, :50]
ad_id = id_embeddings[:, 50:100]
product_id = id_embeddings[:, 100:150]
product_category = id_embeddings[:, 150:200]
advertiser_id= id_embeddings[:, 200:250]
industry = id_embeddings[:, 250:300]

advertiser_id_industry = id_embeddings[:, 300:350]
product_category_advertiser_id = id_embeddings[:, 350:400]
product_category_industry = id_embeddings[:, 400:450]
product_id_advertiser_id = id_embeddings[:, 450:500]
product_id_industry = id_embeddings[:, 500:550]
product_id_product_category = id_embeddings[:, 550:600]

id_embedding_glove_short = np.concatenate([ad_id, creative_id, product_id_product_category, advertiser_id, advertiser_id_industry, product_category_advertiser_id, product_id_advertiser_id], axis=1)
id_embedding_glove_short_pair = np.concatenate([ad_id, creative_id, product_id_product_category, advertiser_id, advertiser_id_industry, product_category_advertiser_id, product_id_advertiser_id, product_id], axis=1)
np.save('%s/id_embedding_glove_short.npy' % my_var_dir, id_embedding_glove_short)
np.save('%s/id_embedding_glove_short_pair.npy' % my_var_dir, id_embedding_glove_short_pair)


# # id_embedding_wv_short

# In[8]:


id_embeddings = np.load('%s/id_embedding50.npy' % my_var_dir)
creative_id = id_embeddings[:, :50]
ad_id = id_embeddings[:, 50:100]
product_id = id_embeddings[:, 100:150]
product_category = id_embeddings[:, 150:200]
advertiser_id= id_embeddings[:, 200:250]
industry = id_embeddings[:, 250:300]

advertiser_id_industry = id_embeddings[:, 300:350]
product_category_advertiser_id = id_embeddings[:, 350:400]
product_category_industry = id_embeddings[:, 400:450]
product_id_advertiser_id = id_embeddings[:, 450:500]
product_id_industry = id_embeddings[:, 500:550]
product_id_product_category = id_embeddings[:, 550:600]
id_embedding_wv_short = np.concatenate([ad_id, creative_id, product_id_product_category, advertiser_id, advertiser_id_industry, product_category_advertiser_id, product_id_advertiser_id], axis=1)
np.save('%s/id_embedding_wv_short.npy'%my_var_dir, id_embedding_wv_short)


# # id_embedding_glove_100

# In[9]:


logging.info('load')
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'%my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'%my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_creative_id_industry.pickle'%my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_creative_id_product_category.pickle'%my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_creative_id_product_id.pickle'%my_var_dir, 'rb')),
}
logging.info('finish load')
mp_creative_other['product_id'] = mp_creative_other['product_id'].replace("\\N", 0)
mp_creative_other['industry'] = mp_creative_other['industry'].replace("\\N", 0)
mp_creative_other['product_id'] = mp_creative_other['product_id'].apply(lambda x : int(x))
mp_creative_other['industry'] = mp_creative_other['industry'].apply(lambda x : int(x))

root_path = var_dir

se_creative_id = pickle.load(open('%s/se_glove_creative_id_100dim.pickle' % root_path, 'rb'))
se_glove_ad_id = pickle.load(open('%s/se_glove_ad_id_100dim.pickle' % root_path, 'rb'))
se_glove_product_id = pickle.load(open('%s/se_glove_product_id_100dim.pickle' % root_path, 'rb'))
se_glove_product_category = pickle.load(open('%s/se_glove_product_category_100dim.pickle' % root_path, 'rb'))
se_glove_advertiser_id = pickle.load(open('%s/se_glove_advertiser_id_100dim.pickle' % root_path, 'rb'))
se_glove_industry = pickle.load(open('%s/se_glove_industry_100dim.pickle' % root_path, 'rb'))

se_glove_advertiser_id_industry = pickle.load(open('%s/se_glove_advertiser_id_industry_100dim.pickle' % root_path, 'rb'))
se_glove_product_category_advertiser_id = pickle.load(open('%s/se_glove_product_category_advertiser_id_100dim.pickle' % root_path, 'rb'))
se_glove_product_category_industry = pickle.load(open('%s/se_glove_product_category_industry_100dim.pickle' % root_path, 'rb'))
se_glove_product_id_advertiser_id = pickle.load(open('%s/se_glove_product_id_advertiser_id_100dim.pickle' % root_path, 'rb'))
se_glove_product_id_industry = pickle.load(open('%s/se_glove_product_id_industry_100dim.pickle' % root_path, 'rb'))
se_glove_product_id_product_category = pickle.load(open('%s/se_glove_product_id_product_category_100dim.pickle' % root_path, 'rb'))


# In[ ]:


def se2np(se):
    return np.array(list(se)).astype('float32')
embedding = np.zeros((4445721, 1200)).astype('float32')
if offline == False:

    creative_ids_str = [str(i) for i in range(1, 4445721)]
    creative_ids = list(range(1, 4445721))

    embedding[1:, : 100] = se2np(se_creative_id[creative_ids_str])
    print(1)

    ad_id_str = [ str(item) for item in mp_creative_other['ad_id'][creative_ids].values]
    embedding[1:, 100 : 200] = se2np(se_glove_ad_id[ad_id_str])
    print(2)

    product_id_str = [ str(item) for item in mp_creative_other['product_id'][creative_ids].values]
    embedding[1:, 200 : 300] = se2np(se_glove_product_id[product_id_str])
    print(3)

    product_category_str = [ str(item) for item in mp_creative_other['product_category'][creative_ids].values]
    embedding[1:, 300 : 400] = se2np(se_glove_product_category[product_category_str])
    print(4)

    advertiser_id_str = [ str(item) for item in mp_creative_other['advertiser_id'][creative_ids].values]
    embedding[1:, 400 : 500] = se2np(se_glove_advertiser_id[advertiser_id_str])
    print(5)

    industry_str = [ str(item) for item in mp_creative_other['industry'][creative_ids].values]
    embedding[1:, 500 : 600] = se2np(se_glove_industry[industry_str])
    print(6)

    advertiser_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(advertiser_id_str, industry_str)]
    embedding[1:, 600 : 700] = se2np(se_glove_advertiser_id_industry[advertiser_id_industry_str])
    print(7)

    product_category_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, advertiser_id_str)]
    embedding[1:, 700 : 800] = se2np(se_glove_product_category_advertiser_id[product_category_advertiser_id_str])
    print(8)

    product_category_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, industry_str)]
    embedding[1:, 800 : 900] = se2np(se_glove_product_category_industry[product_category_industry_str])
    print(9)

    product_id_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, advertiser_id_str)]
    embedding[1:, 900 : 1000] = se2np(se_glove_product_id_advertiser_id[product_id_advertiser_id_str])
    print(10)

    product_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, industry_str)]
    embedding[1:, 1000 : 1100] = se2np(se_glove_product_id_industry[product_id_industry_str])
    print(11)

    product_id_product_category_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, product_category_str)]
    embedding[1:, 1100 : 1200] = se2np(se_glove_product_id_product_category[product_id_product_category_str])
    print(12)

logging.info('start save id embedding')
np.save('var/id_embedding_glove_100.npy', embedding)
logging.info('finish save id embedding')


# # id_embedding_glove_200acc

# In[5]:


glove_fjw = np.load('%s/glove_model_200dim.npy' % my_var_dir)
glove_100 = np.load('%s/id_embedding_glove_100.npy' % my_var_dir)


# In[6]:


creative_id = glove_fjw[:, :200]
ad_id = glove_fjw[:, 200:400]
advertiser_id = glove_fjw[:, 600:800]


# In[7]:


product_id_product_category = glove_100[:, 1100 : 1200]
advertiser_id_industry = glove_100[:, 600: 700]
product_category_advertiser_id = glove_100[:, 700:800]
product_id_advertiser_id = glove_100[:, 900:1000]


# In[8]:


id_embedding_glove_200acc = np.concatenate([creative_id, ad_id, advertiser_id, advertiser_id_industry, product_category_advertiser_id, 
                product_id_advertiser_id, product_id_product_category], axis=1)


# In[9]:


logging.info('start save id embedding')
np.save('%s/id_embedding_glove_200acc.npy'%my_var_dir, id_embedding_glove_200acc)
logging.info('finish save id embedding')


# # id_embedding_glove100origin

# In[ ]:


id_embeddings = np.load('%s/id_embedding_glove_100.npy'%my_var_dir)
np.save('%s/id_embedding_glove100origin.npy'%my_var_dir, id_embeddings[:, :600])


# # id_embedding_wv100

# In[6]:


logging.info('load wv')
wv_model_dict = {
    'creative_id' : pickle.load(open('%s/wv_creative_id_sg_window175_dim100.pickle'%var_dir, 'rb')),
    'ad_id' : pickle.load(open('%s/wv_ad_id_sg_window175_dim100.pickle'%var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/wv_product_id_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_category' : pickle.load(open('%s/wv_product_category_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'advertiser_id' :  pickle.load(open('%s/wv_advertiser_id_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'industry' : pickle.load(open('%s/wv_industry_sg_window175_dim100.pickle'%var_dir, 'rb')),
    
    'advertiser_id_industry' : pickle.load(open('%s/cross_wv_model_advertiser_id_industry_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_category_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_category_advertiser_id_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_category_industry' : pickle.load(open('%s/cross_wv_model_product_category_industry_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_id_advertiser_id' : pickle.load(open('%s/cross_wv_model_product_id_advertiser_id_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_id_industry' : pickle.load(open('%s/cross_wv_model_product_id_industry_sg_window175_dim100.pickle'%var_dir, 'rb')), 
    'product_id_product_category' : pickle.load(open('%s/cross_wv_model_product_id_product_category_sg_window175_dim100.pickle'%var_dir, 'rb')), 
}
logging.info('finish wv')

logging.info('load')
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'%my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'%my_var_dir, 'rb')),
    'industry' : pickle.load(open('%s/se_creative_id_industry.pickle'%my_var_dir, 'rb')),
    'product_category' : pickle.load(open('%s/se_creative_id_product_category.pickle'%my_var_dir, 'rb')),
    'product_id' : pickle.load(open('%s/se_creative_id_product_id.pickle'%my_var_dir, 'rb')),
}
logging.info('finish load')

mp_creative_other['product_id'] = mp_creative_other['product_id'].replace("\\N", 0)
mp_creative_other['industry'] = mp_creative_other['industry'].replace("\\N", 0)
mp_creative_other['product_id'] = mp_creative_other['product_id'].apply(lambda x : int(x))
mp_creative_other['industry'] = mp_creative_other['industry'].apply(lambda x : int(x))


# In[7]:


# np.save('%s/id_embedding_wv100.npy'%my_var_dir,  np.zeros((4445721, 1200)).astype('float32'))


# In[ ]:


embedding = np.zeros((4445721, 1200)).astype('float32')
if offline == False:

    creative_ids_str = [str(i) for i in range(1, 4445721)]
    creative_ids = list(range(1, 4445721))

    embedding[1:, : 100] = wv_model_dict['creative_id'].wv[creative_ids_str]

    ad_id_str = [ str(item) for item in mp_creative_other['ad_id'][creative_ids].values]
    embedding[1:, 100 : 200] = wv_model_dict['ad_id'].wv[ad_id_str]

    product_id_str = [ str(item) for item in mp_creative_other['product_id'][creative_ids].values]
    embedding[1:, 200 : 300] = wv_model_dict['product_id'].wv[product_id_str]

    product_category_str = [ str(item) for item in mp_creative_other['product_category'][creative_ids].values]
    embedding[1:, 300 : 400] = wv_model_dict['product_category'].wv[product_category_str]

    advertiser_id_str = [ str(item) for item in mp_creative_other['advertiser_id'][creative_ids].values]
    embedding[1:, 400 : 500] = wv_model_dict['advertiser_id'].wv[advertiser_id_str]

    industry_str = [ str(item) for item in mp_creative_other['industry'][creative_ids].values]
    embedding[1:, 500 : 600] = wv_model_dict['industry'].wv[industry_str]


    advertiser_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(advertiser_id_str, industry_str)]
    embedding[1:, 600 : 700] = wv_model_dict['advertiser_id_industry'].wv[advertiser_id_industry_str]

    product_category_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, advertiser_id_str)]
    embedding[1:, 700 : 800] = wv_model_dict['product_category_advertiser_id'].wv[product_category_advertiser_id_str]

    product_category_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_category_str, industry_str)]
    embedding[1:, 800 : 900] = wv_model_dict['product_category_industry'].wv[product_category_industry_str]

    product_id_advertiser_id_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, advertiser_id_str)]
    embedding[1:, 900 : 1000] = wv_model_dict['product_id_advertiser_id'].wv[product_id_advertiser_id_str]

    product_id_industry_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, industry_str)]
    embedding[1:, 1000 : 1100] = wv_model_dict['product_id_industry'].wv[product_id_industry_str]

    product_id_product_category_str = [str(a) + "_" + str(b) for (a, b) in zip(product_id_str, product_category_str)]
    embedding[1:, 1100 : 1200] = wv_model_dict['product_id_product_category'].wv[product_id_product_category_str]

logging.info('start save id embedding')
np.save('%s/id_embedding_wv100.npy'%my_var_dir, embedding)
logging.info('finish save id embedding')


# # id_embedding_aa100_wv_glove

# In[ ]:


id_embedding_wv = np.load('%s/id_embedding_wv100.npy'%my_var_dir)
id_embedding_glove = np.load('%s/id_embedding_glove_100.npy'%my_var_dir)
id_embedding_wv_glove_graph = np.concatenate([id_embedding_wv, id_embedding_glove], axis = 1)
id_embedding_wv_glove_graph = id_embedding_wv_glove_graph.astype('float32')
logging.info('start save id embedding')
np.save('%s/id_embedding_wv_glove_graph.npy'%my_var_dir, id_embedding_wv_glove_graph)
logging.info('finish save id embedding')


# In[5]:


# np.save('%s/id_embedding_aa100_wv_glove'%my_var_dir, np.zeros((4445721, 400)))


# In[ ]:


id_embedding_wv_glove_graph = np.load('%s/id_embedding_wv_glove_graph.npy'%my_var_dir)
ad_id_wv = id_embedding_wv_glove_graph[:, 100:200]
advertiser_id_wv = id_embedding_wv_glove_graph[:, 400:500]

ad_id_glove = id_embedding_wv_glove_graph[:, 1200+100:1200+200]
advertiser_id_glove = id_embedding_wv_glove_graph[:, 1200+400:1200+500]
id_embedding_aa100_wv_glove = np.concatenate([ad_id_wv, ad_id_glove, advertiser_id_wv, advertiser_id_glove], axis = 1)
np.save('%s/id_embedding_aa100_wv_glove'%my_var_dir, id_embedding_aa100_wv_glove)


# # id_embedding_aa200_epoch16

# In[4]:


logging.info('load wv')
wv_model_dict = {
    'ad_id' : pickle.load(open('%s/wv_ad_id_sg_window175_dim200_epoch16.pickle'%var_dir, 'rb')),
    'advertiser_id' :  pickle.load(open('%s/wv_advertiser_id_sg_window175_dim200_epoch16.pickle'%var_dir, 'rb')), 
}
logging.info('load')
mp_creative_other = {
    'ad_id' : pickle.load(open('%s/se_creative_id_ad_id.pickle'%my_var_dir, 'rb')),
    'advertiser_id' : pickle.load(open('%s/se_creative_id_advertiser_id.pickle'%my_var_dir, 'rb')),
}
logging.info('finish load')


# In[5]:


np.save('%s/id_embedding_aa200_epoch16.npy'%my_var_dir, np.zeros((4445721, 400)).astype('float32'))


# In[ ]:


embedding = np.zeros((4445721, 400)).astype('float32')
creative_ids_str = [str(i) for i in range(1, 4445721)]
creative_ids = list(range(1, 4445721))
ad_id_str = [ str(item) for item in mp_creative_other['ad_id'][creative_ids].values]
embedding[1:,  : 200] = wv_model_dict['ad_id'].wv[ad_id_str]
advertiser_id_str = [ str(item) for item in mp_creative_other['advertiser_id'][creative_ids].values]
embedding[1:, 200 : 400] = wv_model_dict['advertiser_id'].wv[advertiser_id_str]

logging.info('start save id embedding')
np.save('%s/id_embedding_aa200_epoch16.npy'%my_var_dir, embedding)
logging.info('finish save id embedding')

