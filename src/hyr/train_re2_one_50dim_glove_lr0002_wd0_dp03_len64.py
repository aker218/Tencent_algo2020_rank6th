#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
f = F
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
# import seaborn as sns

import IPython.display as ipd
import copy
import random
# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize(progress_bar=True)
# df.parallel_apply(func)
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold

from transformers import *
import torch.nn as nn
import math
from tqdm.notebook import tqdm 
from transformers.modeling_bert import BertConfig, BertEncoder, BertAttention,BertSelfAttention,BertLayer,BertPooler,BertLayerNorm


# In[2]:


var_dir = '../../var'
my_var_dir = '../../var/hyr'


# # Read

# In[3]:


logging.info('start read df data')

df_train_user = pd.read_csv('%s/data/train_semi_final/user.csv'% var_dir)
pre_df_train_user = pd.read_csv('%s/data/train_preliminary/user.csv'% var_dir)
df_train_user = pd.concat([pre_df_train_user, df_train_user])

train_user = list(range(1, 3000001))
test_user = list(range(3000001, 4000001))

train_gender = np.array(list(df_train_user['gender']))
train_age = np.array(list(df_train_user['age']))
id_embedding_feq = np.load('%s/id_embedding_glove_short.npy'%my_var_dir)
se_tfidf_stack = pickle.load(open('%s/se_tfidf_stack_new.pickle'%my_var_dir, 'rb'))
df_user_info = pickle.load(open('%s/df_user_info_shuffle.pickle'%my_var_dir, 'rb'))
target_encode_user_dict, mp_target_encode = pickle.load(open('%s/target_info_new.pickle'%my_var_dir, 'rb'))

logging.info('finish read df data')


# In[ ]:


offline = df_train_user.shape[0] < 30000
if offline:
    train_user = list(range(1, 301))
    test_user = list(range(3000001, 3000101))


# # 调参

# In[31]:


from collections import namedtuple

ARG = namedtuple('ARG', [
    'batch_size',
    'epoch', 
    'lr',
    'weight_decay',
    'debug',
    'n_embedding',
    'max_length',
    'n_eval',
    'n_worker',
    'device',
    
    'n_gpu',
    'card_list',
    
    'n_fold',
    'save_path',
])
 
args = ARG(
    batch_size = 64 if offline else 256,
    epoch = 10,
    lr = 0.002,
    weight_decay = 0.,
    debug = False,
    n_embedding = 100,
    max_length = 64,
    n_eval = 100000,
    n_worker = 2,
    device=torch.device("cuda:1"),
#     device=torch.device("cpu"),

    n_gpu = 2,
    card_list = [0, 1],
    
    n_fold = 5,
    save_path = '%s/model_one_50dim_glove_lr0002_wd0_dp03_len64/' % my_var_dir,
    
    
)

if args.debug:
    debug_number = 2000
    sub_train_user = train_user[:debug_number]
    sub_train_gender = train_gender[:debug_number] - 1
    sub_train_age = train_age[:debug_number] - 1
    sub_test_user = test_user[:debug_number]
else:
    sub_train_user = train_user
    sub_train_gender = train_gender - 1
    sub_train_age = train_age - 1
    sub_test_user = test_user


# # dataset

# In[32]:


class AdDataset(Data.Dataset):
    def __init__(self, user_ids, gender = None, age = None):
        self.user_id = list(user_ids)
        self.gender = gender if gender is not None else [-1 for _ in range(len(self.user_id))]
        self.age = age if age is not None else [-1 for _ in range(len(self.user_id))]
        
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self,idx):
        return [self.user_id[idx], self.gender[idx], self.age[idx]]
    
feature_name = ['ad_id', 'product_category', 'advertiser_id', 'industry']

n_embedding = 50
x_dict = {
    'time' :  np.zeros((args.batch_size, args.max_length)).astype('int'),
    'click_time' :  np.zeros((args.batch_size, args.max_length)).astype('int'),

    'creative_id' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'ad_id' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_id' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'), 
    'product_category' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'), 
    'advertiser_id' :  np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32') , 
    'industry' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    
    'advertiser_id_industry' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_category_advertiser_id' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_category_industry' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_id_advertiser_id' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_id_industry' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    'product_id_product_category' : np.zeros((args.batch_size, args.max_length, n_embedding)).astype('float32'),
    
    'x_len' :  np.zeros((args.batch_size,)).astype('long'),
}


x_target_encode = np.zeros((args.batch_size, args.max_length, 72)).astype('float32')
def collate_fn(samples):
    sample_np = np.array(samples)
    user_ids = sample_np[:, 0]
    gender = sample_np[:, 1]
    age = sample_np[:, 2]
    
    
    for i, user in enumerate(user_ids):
        S = time.time()
        df_user_info_sub = df_user_info.loc[user]
        if(len(df_user_info_sub['creative_id']) > args.max_length):
            random_idx = random.sample(range(len(df_user_info_sub['creative_id'])), args.max_length)
            random_idx = sorted(random_idx)
            creative_ids = np.array(df_user_info_sub['creative_id'])[random_idx].tolist()
        else:
            creative_ids = df_user_info_sub['creative_id']
        
        len_data = len(creative_ids)
        
        x_dict['time'][i][:len_data] = df_user_info_sub['time'][:len_data]
        x_dict['click_time'][i][:len_data] = df_user_info_sub['click_time'][:len_data]
                
        flod = target_encode_user_dict[user]
        x_target_encode[i][:len_data] = np.array(list(mp_target_encode[flod].loc[creative_ids]))
        
        id_embeddings = id_embedding_feq[creative_ids]
        x_dict['ad_id'][i][:len_data] = id_embeddings[:, :50]
        x_dict['creative_id'][i][:len_data] = id_embeddings[:, 50:100]
        x_dict['product_id_product_category'][i][:len_data] = id_embeddings[:, 100:150]
        x_dict['advertiser_id'][i][:len_data] = id_embeddings[:, 150:200]
        x_dict['advertiser_id_industry'][i][:len_data] = id_embeddings[:, 200:250]
        x_dict['product_category_advertiser_id'][i][:len_data] = id_embeddings[:, 250:300]
        x_dict['product_id_advertiser_id'][i][:len_data] = id_embeddings[:, 300:350]

        x_dict['x_len'][i] = len_data
        

    len_user = sample_np.shape[0]
    
#     cat_feature = torch.cat([torch.tensor(x_dict['ad_id'][:len_user]), torch.tensor(x_dict['product_category'][:len_user]),
#                             torch.tensor(x_dict['advertiser_id'][:len_user]), torch.tensor(x_dict['industry'][:len_user])], dim = 2)
    

    if gender[0] == -1:
        gender = None
        age = None
    else:
        gender = torch.tensor(gender)
        age = torch.tensor(age)
    
    
    return {
        'time' : torch.tensor(x_dict['time'][:len_user]).long(),
        'click_time' : torch.tensor(x_dict['click_time'][:len_user]).long(),
        
        'creative_id' : torch.tensor(x_dict['creative_id'][:len_user]),
        'ad_id' : torch.tensor(x_dict['ad_id'][:len_user]),
        'advertiser_id' : torch.tensor(x_dict['advertiser_id'][:len_user]), 
        
        'advertiser_id_industry' : torch.tensor(x_dict['advertiser_id_industry'][:len_user]),
        'product_category_advertiser_id' : torch.tensor(x_dict['product_category_advertiser_id'][:len_user]),
        'product_id_advertiser_id' : torch.tensor(x_dict['product_id_advertiser_id'][:len_user]),
        'product_id_product_category' : torch.tensor(x_dict['product_id_product_category'][:len_user]),
        

        'target_encode_sequence' : torch.tensor(x_target_encode[:len_user]).float(),
        
        'x_len' :  torch.tensor(x_dict['x_len'][:len_user]),
        'x_flatten' : torch.tensor(list(se_tfidf_stack[user_ids].values)).float(),
        'gender' : gender,
        'age' : age,
        }


# # train

# In[33]:


TIME_FORWARD = 0
TIME_BACKWARD = 0
    

def predict_batch_multi_task(model, user_ids, batch_size = args.batch_size):
    len_user_ids = len(user_ids)
    pre_list_gender = []
    pre_list_age = []
    pre_list_hidden = []
    
    train_dataset=AdDataset(user_ids)
    data_loader = Data.DataLoader(
        dataset=train_dataset,      
        batch_size=args.batch_size,      
        shuffle=False,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader)):
            pre_gender, pre_age, pre_hidden = model(**data)
            pre_list_gender.append(pre_gender.cpu().detach().numpy())
            pre_list_age.append(pre_age.cpu().detach().numpy())      
            pre_list_hidden.append(pre_hidden.cpu().detach().numpy())
            
    return {
        'gender' : np.concatenate(pre_list_gender), 
        'age' : np.concatenate(pre_list_age),
    }

def eval_data(model, user_ids, gender_labels, age_labels):
    choose_idx = list(range(len(user_ids)))
    if(len(user_ids) > args.n_eval):
        choose_idx = random.sample(choose_idx, args.n_eval)
    ret_dict = predict_batch_multi_task(model, user_ids[choose_idx])

    predict_gender = np.argmax(ret_dict['gender'], axis = 1)
    predict_age = np.argmax(ret_dict['age'], axis = 1)
    acc_gender = accuracy_score(gender_labels[choose_idx], predict_gender)
    acc_age = accuracy_score(age_labels[choose_idx], predict_age)
    return acc_gender, acc_age


best_score = 0
def train_multi_task(n_fold, model_class, class_parms, train_dataset, val_dataset, test_user_id):
    
    global TIME_FORWARD, TIME_BACKWARD, best_score
    best_score = 0

    train_user_id = train_dataset['x']
    train_gender = train_dataset['gender']
    train_age = train_dataset['age']
    
    logging.info('train number %d, val number %d' % (len(train_user_id), len(val_dataset['x'])))
    
    torch_dataset = AdDataset(train_user_id, train_gender, train_age)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,      
        batch_size=args.batch_size,      
        shuffle=True,
        collate_fn=collate_fn,
        num_workers = args.n_worker,
    )
    
    model = model_class(**class_parms).to(args.device)
            
    
    no_decay = ["bias", "gamma","beta"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(train_user_id)//(args.batch_size)), num_training_steps=int(len(train_user_id) / args.batch_size * args.epoch)
    )

    for epoch in range(args.epoch):
        loss_list, loss_gender_list, loss_age_list = [], [], []
        model.train()
        
        for step, data in enumerate(tqdm(data_loader)):
            #forward
            S = time.time()

            loss, loss_gender, loss_age, pre_gender, pre_age, _ = model(**data)
        
            TIME_FORWARD += time.time() - S
            
            loss_list.append(float(loss))
            loss_gender_list.append(float(loss_gender))
            loss_age_list.append(float(loss_age))
            
            #backward
            S = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5)
            optimizer.step()
            scheduler.step()
            TIME_BACKWARD += time.time() - S
        
        model.eval()

        train_acc_gender, train_acc_age = eval_data(model, train_dataset['x'], train_dataset['gender'], train_dataset['age'])

        val_acc_gender, val_acc_age = eval_data(model, val_dataset['x'], val_dataset['gender'], val_dataset['age'])
        
        if(val_acc_gender + val_acc_age > best_score):
            torch.save(model, '%s/model_%d_best' % (args.save_path, n_fold))
            best_score = val_acc_gender + val_acc_age

        
        logging.info('forward:%f backward:%f'%(TIME_FORWARD,TIME_BACKWARD))
        logging.info("flod %d epoch %d : \n loss: %f loss_gender : %f, loss_age : %f, gender : %f, %f, age : %f, %f, score:%f" %                     (n_fold, epoch, np.mean(loss_list), np.mean(loss_gender_list), np.mean(loss_age_list),                       train_acc_gender, val_acc_gender, train_acc_age, val_acc_age, val_acc_gender + val_acc_age))
    
    
    if(best_score - val_acc_gender - val_acc_age > 0.00001):
        model = torch.load('%s/model_%d_best' % (args.save_path, n_fold))
        
    val_ret_dict = predict_batch_multi_task(model, val_dataset['x'])
    test_ret_dict = predict_batch_multi_task(model, test_user_id)
    
    return model, val_ret_dict, test_ret_dict


def nn_cross_validation_multi_task(x_train, gender, age, x_test, model_class, class_parms, func_train, is_cross = True, random_seed = 0):
    
    folds = KFold(n_splits=args.n_fold, shuffle=False)
    
    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)
    
    x_train_val = np.array(x_train)
    gender_train_val = np.array(gender)
    age_train_val = np.array(age)


    score_gender_val = np.zeros((len(x_train), 2))
    score_age_val = np.zeros((len(x_train), 10))
    

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, gender_train_val)):
        save_trn_idx = trn_idx
        save_val_idx = val_idx
        
        train_x, train_gender, train_age = x_train_val[trn_idx], gender_train_val[trn_idx], age_train_val[trn_idx]
        val_x, val_gender, val_age = x_train_val[val_idx], gender_train_val[val_idx], age_train_val[val_idx]
        train_dataset = {
            'x' : train_x,
            'gender' : train_gender,
            'age' : train_age,
        }
        val_dataset = {
            'x' : val_x,
            'gender' : val_gender,
            'age' : val_age
        }
        
        model, val_ret_dict, test_ret_dict = func_train(n_fold, model_class, class_parms, train_dataset, val_dataset, x_test)
        
        score_gender_val[val_idx] = val_ret_dict['gender']
        score_age_val[val_idx] = val_ret_dict['age']
        
        val_predict_gender = np.argmax(score_gender_val[val_idx] , axis = 1)
        val_predict_age = np.argmax(score_age_val[val_idx] , axis = 1)

        if is_cross == False:
            eda_val_dict = {
                'user' : val_x,
                'pre_gender' : val_predict_gender,
                'gender' : val_gender,
                'pre_age' : val_predict_age,
                'age' : val_age,
                'score_gender' : score_gender_val[val_idx],
                'score_age' :  score_age_val[val_idx]
            }

            return model, eda_val_dict, test_ret_dict
        
        acc_gender = accuracy_score(val_gender, val_predict_gender)
        acc_age = accuracy_score(val_age, val_predict_age)

        logging.info("%d : gender : %f, age: %f, score :%f" % (n_fold, acc_gender, acc_age, acc_gender + acc_age))

        torch.save(model, '%s/model_%d' % (args.save_path, n_fold))
        pickle.dump(test_ret_dict, open('%s/test_dict_flod_%d.pickle' % (args.save_path, n_fold), 'wb'))
        
        test_gender_pre = np.argmax(test_ret_dict['gender'], axis = 1) + 1
        test_age_pre = np.argmax(test_ret_dict['age'], axis = 1) + 1
        df_submit = pd.DataFrame()
        df_submit['user_id'] = x_test
        df_submit['predicted_gender'] = test_gender_pre
        df_submit['predicted_age'] = test_age_pre
        df_submit.to_csv('%s/df_submit_flod_%d.csv' % (args.save_path, n_fold), index=False)
        
        
    return score_gender_val, score_age_val


# # Model

# In[34]:


class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,X):
        mean=X.mean(-1,keepdim=True)
        std=X.std(-1,keepdim=True)
        return self.gamma*(X-mean)/(std+self.eps)+self.beta
    
class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        assert all(k % 2 == 1 for k in kernel_sizes), 'only support odd kernel sizes'
        assert out_channels % len(kernel_sizes) == 0, 'out channels must be dividable by kernels'
        out_channels = out_channels // len(kernel_sizes)
        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
            nn.init.normal_(conv.weight, std=math.sqrt(2. / (in_channels * kernel_size)))
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), GeLU()))
        self.model = nn.ModuleList(convs)

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.model], dim=-1)

class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.encoders = nn.ModuleList([Conv1d(
                in_channels=input_size if i == 0 else args.hidden_size,
                out_channels=args.hidden_size,
                kernel_sizes=args.kernel_sizes) for i in range(args.enc_layers)])

    def forward(self, x, mask):
        x = x.transpose(1, 2)  # B x C x L
        mask = mask.transpose(1, 2)
        for i, encoder in enumerate(self.encoders):
            x.masked_fill_(~mask, 0.)
            if i > 0:
                x = f.dropout(x, self.dropout, self.training)
            x = encoder(x)
        x = f.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)  # B x L x C
    
class FullFusion(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.dropout = args.dropout
        self.fusion1 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, args.hidden_size, activations=True)
        self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)
    
class AugmentedResidual(nn.Module):
    def forward(self, x, res, i):
        if i == 1:
            return torch.cat([x, res], dim=-1)  # res is embedding
        hidden_size = x.size(-1)
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding
    
class Pooling(nn.Module):
    def forward(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]

class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class Alignment(nn.Module):
    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))
        self.summary = {}

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature
    
    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b)
        return feature_a, feature_b

class MappedAlignment(Alignment):
    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(input_size, args.hidden_size, activations=True),
        )

    def _attention(self, a, b):
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)

class AlignmentOne(nn.Module):
    def __init__(self, args, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(args.hidden_size)))
        self.summary = {}

    def _attention(self, a):
        return torch.matmul(a, a.transpose(1, 2)) * self.temperature
    

    def forward(self, a, mask_a):
        attn = self._attention(a)
#         mask = torch.matmul(mask_a.float(), mask_a.transpose(1, 2).float()).byte()
        mask = mask_a.byte()
        attn.masked_fill_(~mask, -1e7)
        attn_a = f.softmax(attn, dim=1)
        feature_a = torch.matmul(attn_a.transpose(1, 2), a)
        return feature_a

class RE2One(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': AlignmentOne(
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': FullFusion(
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = AugmentedResidual()
        self.pooling = Pooling()
            
    def make_mask(self, X, valid_len):
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
        return mask.unsqueeze(2).byte()

    def forward(self, a, x_len):
        mask_a = self.make_mask(a, x_len)
        res_a = a
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                res_a = a
            a_enc = block['encoder'](a, mask_a)
            a = torch.cat([a, a_enc], dim=-1)
            align_a = block['alignment'](a, mask_a)
            a = block['fusion'](a, align_a)        
        

        hidden= self.pooling(a, mask_a)
        
        
        
        return hidden
    
class RE2Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': MappedAlignment(
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': FullFusion(
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = AugmentedResidual()
        self.pooling = Pooling()
        
        
#         self.ln_100s = nn.ModuleList([LayerNorm(100) for _ in range(24)])
    
    def make_mask(self, X, valid_len):
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
        return mask.unsqueeze(2).byte()
       

    def forward(self, a, b, x_len):
        mask_a = self.make_mask(a, x_len)
        mask_b = self.make_mask(b, x_len)
        
        res_a, res_b = a, b
        
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        
        

        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        
        hidden = torch.cat([a, b, (a - b).abs(), a * b], dim=-1) #symmetric
        
        return hidden

class RE2(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_dim = 50
        self.ln_tfidf_stack = LayerNorm(55)
        self.ln_one_hot_target = LayerNorm(72)

        n_time_embedding = 16
        self.position_embeddings = nn.Embedding(92, n_time_embedding)
        self.click_time_embeddings = nn.Embedding(33, n_time_embedding)

        self.ln_50s = nn.ModuleList([LayerNorm(50) for _ in range(14)])
        
        
        n_cat_hidden = 1
        n_flatten = 55
        self.decoder_gender = nn.Sequential(Linear( n_cat_hidden * args.hidden_size + n_flatten, 2))
        self.decoder_age = nn.Sequential(Linear(n_cat_hidden * args.hidden_size + n_flatten, 10))   
        

        self.ln_hidden = LayerNorm(n_cat_hidden * args.hidden_size)
        self.ln_tfidf_stack = LayerNorm(55)
        self.ln_target_agg = LayerNorm(360)

        
        self.re2_one = RE2One(args)


    def forward(self,
                time,
                click_time,
                
                creative_id,
                ad_id,
                advertiser_id,                
                advertiser_id_industry,
                product_category_advertiser_id,
                product_id_advertiser_id,
                product_id_product_category,
                
                target_encode_sequence,
                x_len,
                x_flatten,
                gender = None,\
                age = None):
        
        
        time = time.to(args.device)
        click_time = click_time.to(args.device)
        
        creative_id = creative_id.to(args.device) 
        ad_id = ad_id.to(args.device) 
        advertiser_id = advertiser_id.to(args.device) 
        advertiser_id_industry = advertiser_id_industry.to(args.device)
        product_category_advertiser_id = product_category_advertiser_id.to(args.device) 
        product_id_advertiser_id = product_id_advertiser_id.to(args.device)
        product_id_product_category = product_id_product_category.to(args.device) 
        
        target_encode_sequence = target_encode_sequence.to(args.device) 
        x_len = x_len.to(args.device) 
        x_flatten = x_flatten.to(args.device)
        
        if gender is not None:
            gender = gender.to(args.device) 
            age = age.to(args.device) 
    
        
        a_wv = torch.cat([
            self.ln_50s[0](ad_id),
            self.ln_50s[1](creative_id),
            self.ln_50s[2](product_id_product_category),
            self.ln_50s[3](advertiser_id),
            self.ln_50s[4](advertiser_id_industry),
            self.ln_50s[5](product_category_advertiser_id),
            self.ln_50s[6](product_id_advertiser_id),

            self.ln_one_hot_target(target_encode_sequence),
            self.position_embeddings(time),
            self.click_time_embeddings(click_time),
        ],dim=-1)

        
        hidden = self.re2_one(a_wv, x_len)
        cat = torch.cat([self.ln_hidden(hidden), self.ln_tfidf_stack(x_flatten)], dim = -1)
        output_age = self.decoder_age(cat)
        output_gender = self.decoder_gender(cat)

        if(gender is None):
            return output_gender, output_age, hidden
        
        
        loss_age = nn.CrossEntropyLoss()
        loss_gender = nn.CrossEntropyLoss()
        l_age = loss_age(output_age,age.long())        
        l_gender = loss_gender(output_gender,gender.long())
            
        
        l=0.5*l_gender + 0.5*l_age
        
        
        return l,l_gender,l_age,output_gender, output_age, hidden


RE2_ARG = namedtuple('ARG', [
    'dropout',
    'hidden_size',
    'enc_layers',
    'kernel_sizes',
    'blocks',
    'embedding_dim',
    'device',
])

re2_args = RE2_ARG(
    dropout = 0.3,
    hidden_size = 256,
    enc_layers = 2,
    kernel_sizes = (3,),
    blocks = 2,
    embedding_dim = 350 + 72 + 16 + 16,
    device = args.device,
)

# model = RE2(re2_args)
# print(sum(param.numel() for param in model.parameters()))

logging.info('start training ')
score_gender_val, score_age_val = nn_cross_validation_multi_task(sub_train_user, sub_train_gender, sub_train_age,                                                                    sub_test_user,                                                                     RE2, {'args' : re2_args}, train_multi_task, True)

pickle.dump(score_gender_val, open('%s/score_gender_val.pickle' % args.save_path, 'wb'))
pickle.dump(score_age_val, open('%s/score_age_val.pickle' % args.save_path, 'wb'))
# logging.info('finish training ')


# In[ ]:




