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
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import IPython.display as ipd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,KFold
import pandas as pd
import numpy as np
import copy
from scipy.special import softmax
from torchcontrib.optim import SWA


# In[2]:


# import os
# paths = []
# names = []
# target_dir = './var/model_ret_dicts/hyr/'
# for filename in os.listdir(target_dir):
#     if(filename[0] == '.'):
#         continue
#     paths.append(target_dir + filename)
#     names.append(filename)
    
# target_dir = './var/model_ret_dicts/fjw/'
# for filename in os.listdir(target_dir):
#     if(filename[0] == '.') :
#         continue
#     paths.append(target_dir + filename)
#     names.append(filename)
# names = list(map(lambda x : x.replace('model_', '') ,names))


# In[3]:


# model_ret_dicts = []
# logging.info("loading feature...") 
# for p in paths:
#     ret = pickle.load(open(p, 'rb'))
#     model_ret_dicts.append(ret)
# logging.info("loading finished...")


# In[4]:


# test_x = np.zeros((1000000, 12 * len(model_ret_dicts))).astype('float32')
# for i in range(len(model_ret_dicts)):
                  
#     test_x[:, i*12:i*12+2] = model_ret_dicts[i]['test_gender']                  
#     test_x[:, i*12+2:i*12+12] = model_ret_dicts[i]['test_age']                  


# In[6]:


# class_20_test_np = np.load('./var/model_ret_dicts/class20_test.npy')
# test_x = np.concatenate([test_x, class_20_test_np], axis=1)
# test_x.shape


# In[2]:


from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import *
import torch.nn as nn


# In[3]:


test_x=np.load("./var/test_x.npy")


# In[4]:



ARG = namedtuple('ARG', [
    'batch_size',
    'epoch',
    'lr',
    'weight_decay',
    'n_worker',
    'device',
    'n_fold'
])
 
args = ARG(
    batch_size = 1024,
    epoch = 10,
    lr = 0.005,
    weight_decay = 0.1,
    n_worker = 0,
    n_fold = 5,
    device=torch.device("cuda:3"),
#     device=torch.device("cpu"),

)


# In[12]:


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_feature = 12 * 32+20
        
        hidden = 324
        out_feature = 256
        self.dense = nn.Sequential(
            nn.Linear(in_feature, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_feature),
        )
        self.decode_gender = nn.Linear(out_feature, 2)
        self.decode_age = nn.Linear(out_feature, 10)
   
    def forward(self, x, gender = None, age = None):
        
        hidden = self.dense(x)
        output_gender = self.decode_gender(hidden)
        output_age = self.decode_age(hidden)
        
        if gender is None:
            return output_gender, output_age
        
        ce = nn.CrossEntropyLoss()
        loss_gender = ce(output_gender, gender.long())
        loss_age = ce(output_age, age.long())
        loss = loss_gender + loss_age
        return loss, loss_gender, loss_age, output_gender, output_age


# In[13]:


def swa(logger, model, model_dir, model_path_list, swa_start):
    """
    :param logger: ...
    :param model: ...
    :param model_dir: ...
    :param model_path_list: this model path list should be increased by steps
    :param swa_start: the epoch when averaging begins. (start with 0)
    :return: model path list extend with swa model
    """

    assert 1 < swa_start <= len(model_path_list) - 1,         f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 1'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(os.path.join(model_dir, _ckpt, 'model.pt'),
                                             map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    swa_model_dir = os.path.join(model_dir, f'checkpoint-swa_start{swa_start}')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    logger.info('Save swa model')

    torch.save(swa_model.state_dict(), os.path.join(swa_model_dir, 'model.pt'))

    model_path_list.append(f'checkpoint-swa_start{swa_start}')

    return model_path_list


# In[14]:


output_dir="../model/"
def predict_batch_multi_task(model, train_x, batch_size = args.batch_size):
    len_user_ids = len(train_x)
    pre_list_gender = []
    pre_list_age = []
    
    train_dataset = Data.TensorDataset(torch.tensor(train_x).float())
    data_loader = Data.DataLoader(
        dataset=train_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        num_workers = args.n_worker,
    )
    with torch.no_grad():
        model.eval()
        for step, data in enumerate(tqdm(data_loader)):
            
            pre_gender, pre_age = model(data[0].to(args.device))
            pre_list_gender.append(pre_gender.cpu().detach().numpy())
            pre_list_age.append(pre_age.cpu().detach().numpy())      
        model.train()
    return {
        'gender' : np.concatenate(pre_list_gender), 
        'age' : np.concatenate(pre_list_age),
    }
    
test_gender = np.zeros((len(test_x), 2))
test_age = np.zeros((len(test_x), 10))
for fold in range(args.n_fold):
    
    model=Dense().to(args.device)
    model.load_state_dict(torch.load("./model/model_"+str(fold+1)+".pt"))
    test_ret_dict=predict_batch_multi_task(model,test_x)
    test_gender += softmax(test_ret_dict['gender'], axis=1) / args.n_fold
    test_age += softmax(test_ret_dict['age'], axis=1) / args.n_fold


# In[15]:


test_gender_pre = np.argmax(test_gender, axis = 1) + 1
test_age_pre = np.argmax(test_age, axis = 1) + 1
df_submit = pd.DataFrame()
df_submit['user_id'] = list(range(3000001, 4000001))
df_submit['predicted_gender'] = test_gender_pre
df_submit['predicted_age'] = test_age_pre
df_submit.to_csv('submission.csv', index=False)

