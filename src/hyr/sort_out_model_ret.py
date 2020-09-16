#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import IPython.display as ipd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
import pandas as pd
import numpy as np
import copy
from scipy.special import softmax


# In[2]:


var_dir = '../../var'
my_var_dir = '../../var/hyr'


# In[7]:


paths = [
        '%s/model_pair_200acc_wv_lr0003_label_smooth' % my_var_dir,
        '%s/model_one_50dim_wv_lr0003_wd0_dp02' % my_var_dir,

    '%s/model_one_50dim_wv' % my_var_dir,
     '%s/model_one_200acc_wv' % my_var_dir,
     '%s/model_one_50dim_glove_origin_lr0003_wd0_dp01_len256' % my_var_dir,
     '%s/model_one_200acc_glove_lr0003_smooth_wd005' % my_var_dir,
     '%s/model_one_50dim_glove_lr0002_wd0_dp03_len64' % my_var_dir,
    
     '%s/model_pair_200acc_glove_lr0003_wd0_dp04' % my_var_dir, 
    '%s/model_pair_200acc_wv_lr0005' % my_var_dir,

    '%s/model_one_50dim_origin100glove_lr0003_wd0_dp03' % my_var_dir, 
     '%s/model_one_50dim_glove_lr0005' % my_var_dir,
    
     '%s/model_re2_pair_50dim_wv' % my_var_dir,
    '%s/model_pair_50dim_glove_smooth_lr0003' % my_var_dir,
    '%s/model_one_aa100_wv_glove_lr0003_wd0_dp03' % my_var_dir,
    '%s/model_pair_aa200_wv_epoch16' % my_var_dir,
    '%s/model_one_200acc_wv_wd0_dp03' % my_var_dir,

]
names = [path.split('/')[-1] for path in paths]

def generate_feature(path, use_softmax):
    n = 10000
#     n = 1
    gender = np.zeros((int(1000000/n), 2))
    age = np.zeros((int(1000000/n), 10))
    
    fold = 5
    for i in range(fold):
        test_ret_dict = pickle.load(open('%s/test_dict_flod_%d.pickle' % (path, i), "rb"))
        if use_softmax:
            gender += softmax(test_ret_dict['gender'], axis = 1) / fold
            age += softmax(test_ret_dict['age'], axis = 1) / fold       
        else:
            gender += test_ret_dict['gender'] / fold
            age += test_ret_dict['age'] / fold       
    
    val_gender = pickle.load(open("%s/score_gender_val.pickle" % path, "rb"))
    val_age = pickle.load(open("%s/score_age_val.pickle" % path, "rb"))
    if use_softmax:
        val_gender = softmax(val_gender, axis=1)
        val_age = softmax(val_age, axis=1)
    return {
        'train_gender' : val_gender,
        'train_age' : val_age,
        'test_gender' : gender,
        'test_age' : age
    }

model_ret_dicts = []
for path in paths:
    model_ret_dict = generate_feature(path, True)
    model_ret_dicts.append(model_ret_dict)
for i in range(len(model_ret_dicts)):
    pickle.dump(model_ret_dicts[i], open('%s/model_ret_dicts/hyr/%s' % (var_dir, names[i]), 'wb'))


# In[8]:


path = '%s/model_one_50dim_200_class/' % my_var_dir
val_list, test_list = [], []
for i in range(5):
    val = pickle.load(open('%s/val_dict_flod_%d.pickle' % (path, i), 'rb'))
    test = pickle.load(open('%s/test_dict_flod_%d.pickle' % (path, i), 'rb'))
    val_list.append(softmax(val, axis=1))
    test_list.append(softmax(test, axis=1))
val_np = np.concatenate(val_list, axis=0)
test_np = np.array(test_list).mean(axis=0)
np.save('%s/model_ret_dicts/class20_val.npy'%var_dir, val_np)
np.save('%s/model_ret_dicts/class20_test.npy'%var_dir, test_np)


# In[ ]:




