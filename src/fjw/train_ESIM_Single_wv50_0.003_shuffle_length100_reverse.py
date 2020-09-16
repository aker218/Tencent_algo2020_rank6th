#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from tqdm import  tqdm_notebook as tqdm
import _pickle as pk
import os
import json
from IPython.display import display,HTML
from category_encoders import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from transformers import *
from transformers.modeling_bert import BertConfig,BertLayerNorm
from transformers.activations import gelu, gelu_new, swish
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import PackedSequence
from torchcontrib.optim import SWA
np.random.seed(13)
import collections
import math
import time
import logging
import copy
logger = logging.getLogger(__name__)
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


# In[2]:


train_features=pk.load(open("../../var/fjw/train_mid/new_feature_simple.pk","rb"))


# In[3]:


full_click_log_time=pk.load(open("../../var/fjw/usr_seq/se_user_time.pickle","rb"))
# full_click_log_creative_id=pk.load(open("./dataset/usr_seq/se_user_creative_id.pickle","rb"))
full_click_log_creative_id=pk.load(open("../../var/fjw/usr_seq/se_user_creative_id_shuffle.pk","rb"))
# full_click_log_click_times=pk.load(open("./dataset/usr_seq/se_user_click_times.pickle","rb"))
full_click_log_click_times=pk.load(open("../../var/fjw/usr_seq/se_user_click_time_shuffle.pk","rb"))


# In[4]:


usr_dict,target_map_dicts=pk.load(open("../../var/fjw/train_mid/new_target_info_simple.pk","rb"))


# In[5]:


tfidf_info=pk.load(open("../../var/fjw/se_tfidf_stack_new.pickle","rb"))
cal_series=pk.load(open("../../var/fjw/simple_cal_norm.csv","rb"))


# In[6]:


w2v_model=np.load("../../var/hyr/id_embedding50.npy")
#glove_model_min_dim50_full.npy
#id_embedding.npy


# In[7]:


label_info=np.array(train_features.values.tolist())
age_nums=[0]*10
gender_nums=[0]*2
for idx in tqdm(range(label_info.shape[0])):
    age_nums[label_info[idx,0]-1]+=1
    gender_nums[label_info[idx,1]-1]+=1
age_nums=np.array(age_nums)/sum(age_nums)
gender_nums=np.array(gender_nums)/sum(gender_nums)
age_nums=torch.tensor(age_nums).float()
gender_nums=torch.tensor(gender_nums).float()


# In[9]:


class UsrDataset(Data.Dataset):
    def __init__(self,examples):
        self.usr_id=examples.index.tolist()
    def __len__(self):
        return len(self.usr_id)
    def __getitem__(self,idx):
        return  self.usr_id[idx]

def collate_fn(usr_ids):
    max_len=100
#     max_len=88  #length 88
        
    train_data=train_features[usr_ids]
    lengths=[]
    times,creative_ids,click_times,ad_ids,product_ids,product_categorys,    advertiser_ids,industrys,target_encodings,cals,tfidfs,ages,genders= [],[],[],[],[],[],[],[],[],[],[],[],[]
    advertiser_id_industrys,    product_cat_advertisers,product_cat_industrys,    product_id_product_cats,product_id_advertisers,product_id_industrys=[],[],[],                                                                            [],[],[]
    tfidf_sequences=[]
    textrank_sequences=[]
    one_hot_targets=[]
    graph_sequences=[]
    target_encode_aggs=torch.zeros((len(usr_ids),360)).float()
    for idx,(usr_id,sample) in enumerate(zip(usr_ids,train_data)):
        if sample[0]>0:
            age,gender=torch.tensor(sample[0]).long(),torch.tensor(sample[1]).long()
        else:
            age,gender=torch.tensor(sample[0]+1).long(),torch.tensor(sample[1]+1).long()
        ages.append(age)
        genders.append(gender)
#         creative_idx=pk.load(open("./dataset/usr_id/%d"%usr_id,"rb"))
        fold=usr_dict[usr_id]
        creative_idx=full_click_log_creative_id.at[usr_id]
        creative_id_list=[str(e) for e in creative_idx]
        time_list=full_click_log_time.at[usr_id]
        times.append(torch.tensor(time_list[-max_len:]).long()-1)
        click_times_list=list(map(lambda x:x if x<=31 else 32,full_click_log_click_times.at[usr_id]))
        click_times.append(torch.tensor(click_times_list[-max_len:]).long()-1)
#         features=np.load(open("./dataset/usr_feature_simple/%d.npy"%usr_id,"rb"))
        one_hot_target=np.stack(target_map_dicts[fold].loc[creative_id_list].values)
        features=one_hot_target
        length=len(creative_idx) if len(creative_idx)<max_len else max_len
        lengths.append(length)
        word2vec=w2v_model[creative_idx]
        cal=cal_series.at[usr_id]
        cals.append(torch.tensor(cal).float())
        tfidf=tfidf_info.at[usr_id]
        tfidfs.append(torch.tensor(tfidf).float())
#         target_encode_aggs[idx]=torch.tensor(target_encode_agg[usr_id]).float()
#         target_encoding=torch.tensor(features[1][:max_len,:12]).float()
#         target_encodings.append(target_encoding)
#         graph_info=torch.tensor(features[:max_len,:28]).float()
#         graph_sequences.append(graph_info)
#         tr_info=torch.tensor(textrank_sequence_info.at[usr_id][:max_len]).float()
#         textrank_sequences.append(tr_info)
#         tfidf_sequence=torch.tensor(tfidf_sequence_info.at[usr_id][:max_len]).float()
#         tfidf_sequences.append(tfidf_sequence)
        one_hot_target=torch.tensor(features[-max_len:,:]).float()
        one_hot_targets.append(one_hot_target)
        
#         creative_id=word2vec[:max_len,list(range(200))]
#         creative_ids.append(torch.tensor(creative_id).float())

#         ad_id=word2vec[:max_len,list(range(200,400))]
#         ad_ids.append(torch.tensor(ad_id).float())

#         product_id=word2vec[:max_len,list(range(400,500))]
#         product_ids.append(torch.tensor(product_id).float())

#         product_category=word2vec[:max_len,list(range(500,600))]
#         product_categorys.append(torch.tensor(product_category).float())

#         advertiser_id=word2vec[:max_len,list(range(600,800))]
#         advertiser_ids.append(torch.tensor(advertiser_id).float())

#         industry=word2vec[:max_len,list(range(800,900))]
#         industrys.append(torch.tensor(industry).float())
    
#         product_cat_industry=word2vec[:max_len,list(range(1000,1100))]
#         product_cat_industrys.append(torch.tensor(product_cat_industry).float())
#         product_id_advertiser=word2vec[:max_len,list(range(900,1000))]
#         product_id_advertisers.append(torch.tensor(product_id_advertiser).float())
  
        creative_id=word2vec[-max_len:,list(range(50))]
        creative_ids.append(torch.tensor(creative_id).float())

        ad_id=word2vec[-max_len:,list(range(50,100))]
        ad_ids.append(torch.tensor(ad_id).float())

        product_id=word2vec[-max_len:,list(range(100,150))]
        product_ids.append(torch.tensor(product_id).float())

        product_category=word2vec[-max_len:,list(range(150,200))]
        product_categorys.append(torch.tensor(product_category).float())

        advertiser_id=word2vec[-max_len:,list(range(200,250))]
        advertiser_ids.append(torch.tensor(advertiser_id).float())

        industry=word2vec[-max_len:,list(range(250,300))]
        industrys.append(torch.tensor(industry).float())
    
#         advertiser_id_industry=word2vec[:max_len,list(range(300,350))]
#         advertiser_id_industrys.append(torch.tensor(advertiser_id_industry).float())
#         product_cat_advertiser=word2vec[:max_len,list(range(350,400))]
#         product_cat_advertisers.append(torch.tensor(product_cat_advertiser).float())
        product_cat_industry=word2vec[-max_len:,list(range(400,450))]
        product_cat_industrys.append(torch.tensor(product_cat_industry).float())
        product_id_advertiser=word2vec[-max_len:,list(range(450,500))]
        product_id_advertisers.append(torch.tensor(product_id_advertiser).float())
#         product_id_industry=word2vec[:max_len,list(range(500,550))]
#         product_id_industrys.append(torch.tensor(product_id_industry).float())
#         product_id_product_cat=word2vec[:max_len,list(range(550,600))]
#         product_id_product_cats.append(torch.tensor(product_id_product_cat).float())

    times=torch.nn.utils.rnn.pad_sequence(times,padding_value=0, batch_first=True)
    creative_ids=torch.nn.utils.rnn.pad_sequence(creative_ids,padding_value=0, batch_first=True)
    click_times=torch.nn.utils.rnn.pad_sequence(click_times,padding_value=0, batch_first=True)
    ad_ids=torch.nn.utils.rnn.pad_sequence(ad_ids,padding_value=0, batch_first=True)
    product_ids=torch.nn.utils.rnn.pad_sequence(product_ids,padding_value=0,batch_first=True)
    product_categorys=torch.nn.utils.rnn.pad_sequence(product_categorys,padding_value=0,batch_first=True)
    advertiser_ids=torch.nn.utils.rnn.pad_sequence(advertiser_ids,padding_value=0,batch_first=True)
    industrys=torch.nn.utils.rnn.pad_sequence(industrys,padding_value=0,batch_first=True)
#     advertiser_id_industrys=torch.nn.utils.rnn.pad_sequence(advertiser_id_industrys,padding_value=0,batch_first=True)
#     product_cat_advertisers=torch.nn.utils.rnn.pad_sequence(product_cat_advertisers,padding_value=0,batch_first=True)
#     product_id_product_cats=torch.nn.utils.rnn.pad_sequence(product_id_product_cats,padding_value=0,batch_first=True)
    product_cat_industrys=torch.nn.utils.rnn.pad_sequence(product_cat_industrys,padding_value=0,batch_first=True)
    product_id_advertisers=torch.nn.utils.rnn.pad_sequence(product_id_advertisers,padding_value=0,batch_first=True)
#     product_id_industrys=torch.nn.utils.rnn.pad_sequence(product_id_industrys,padding_value=0,batch_first=True)
#     target_encodings=torch.nn.utils.rnn.pad_sequence(target_encodings,padding_value=0,batch_first=True)
#     tfidf_sequences=torch.nn.utils.rnn.pad_sequence(tfidf_sequences,padding_value=0,batch_first=True)
#     textrank_sequences=torch.nn.utils.rnn.pad_sequence(textrank_sequences,padding_value=0,batch_first=True)
    one_hot_targets=torch.nn.utils.rnn.pad_sequence(one_hot_targets,padding_value=0,batch_first=True)
#     graph_sequences=torch.nn.utils.rnn.pad_sequence(graph_sequences,padding_value=0,batch_first=True)
#     target_embed_info=one_hot_targets*torch.cat([torch.arange(0,10),torch.arange(0,2)]).repeat(6).float().unsqueeze(0).unsqueeze(0)
#     target_embed_info=[(e.sum(dim=-1)//0.5).long() if idx%2==0  else (e.sum(dim=-1)//0.1).long()\
#                        for idx,e in enumerate(list(torch.split(target_embed_info,\
#                                                                              [10,2,10,2,10,2,10,2,10,2,10,2],dim=-1)))]
    target_embed_info=[torch.zeros(1)]                                                                    
    return torch.tensor(lengths).long(),times,click_times,creative_ids,ad_ids,product_ids,product_categorys,advertiser_ids, industrys,    target_encodings,product_cat_industrys,product_id_advertisers,product_id_industrys,    torch.stack(cals).float(),torch.stack(tfidfs).float(),tfidf_sequences,one_hot_targets,    textrank_sequences,graph_sequences,advertiser_id_industrys,product_cat_advertisers,    product_id_product_cats,target_encode_aggs,torch.stack(ages).long()-1,torch.stack(genders).long()-1


# In[10]:


train_dataset=UsrDataset(train_features)
train_dataloader=Data.DataLoader(train_dataset,batch_size=256,shuffle=False,                                 collate_fn=collate_fn,num_workers=0)


# In[11]:


for e in train_dataloader:
    break


# In[12]:


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if click_times is not None:
            attention_scores=attention_scores*click_times.unsqueeze(1).unsqueeze(1).float()
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,click_times
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,click_times=click_times)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        click_times=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,click_times=click_times
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# In[13]:


class LabelSmoothingCrossEntropy(nn.Module):


    def __init__(self, smoothing=0.1,weights=torch.ones(2)/2):

        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weights=weights.to(device)

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
#         new_target=torch.zeros(X.shape).scatter_(1,target.unsqueeze(1),1)
#         smooth_target=new_target*0.9+torch.ones_like(new_target)*(0.1/new_target.shape[1])
#         -(F.log_softmax(X,dim=-1)*smooth_target).sum(dim=-1).mean()
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = (-logprobs*self.weights).sum(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
def masked_softmax(X,valid_len):
    if valid_len is None:
        return F.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1]).view(-1)
        else:
            valid_len=valid_len.view(-1)
        X=X.view(-1,shape[-1])    
        mask=(torch.arange(0,X.shape[-1]).repeat(X.shape[0],1).float().to(X.device)<valid_len.view(-1,1).repeat(1,X.shape[-1]).float()).float()
        mask=torch.log(mask)
        X=X+mask
        return F.softmax(X,dim=-1).view(shape)
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch
class DotProductAttention(nn.Module):
    def __init__(self,dropout=0):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_len: either (batch_size, ) or (batch_size, xx)
    def forward(self,query,key,value,valid_len=None):
        d=query.shape[-1]
        shape=query.shape
        if valid_len.dim()==1:
            valid_len=valid_len.view(-1,1).repeat(1,shape[1])
        mask=(torch.arange(0,query.shape[1]).repeat(query.shape[0],1).to(query.device)<valid_len).float()
        scores=torch.bmm(query,key.permute(0,2,1))/math.sqrt(d)
        attention_weights=self.dropout(masked_softmax(scores,valid_len))
        return torch.bmm(attention_weights,value)*mask.unsqueeze(-1)
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
class TransEncoder(nn.Module):
    def __init__(self,config):
        super(TransEncoder,self).__init__()
        self.Encoder=BertEncoder(config=config)
        self.P=PositionalEncoding(config)
        self.config=config
        for n,e in self.Encoder.named_modules():
            self._init_weights(e)
        for n,e in self.P.named_modules():
            self._init_weights(e)
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def make_mask(self,X,valid_len):
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def forward(self,X,length):
        #make attention mask
        attention_mask=self.make_mask(X,length)
        embedding_output=self.P(X)
        #adjust attention mask
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #make head mask
        head_mask = [None] * self.config.num_hidden_layers
        outputs=self.Encoder(  embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,)
        return outputs[0]
class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        total_length=sequences_batch.shape[1]
        packed_batch = nn.utils.rnn.pack_padded_sequence(sequences_batch,
                                                         sequences_lengths,
                                                         batch_first=True,enforce_sorted=False)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True,total_length=total_length)

        return outputs
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
class Conditional_LayerNorm(nn.Module):
    def __init__(self,features,conditional_dim,eps=1e-6):
        super(Conditional_LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        self.trans_gamma=nn.Linear(conditional_dim,self.gamma.shape[-1])
        self.trans_beta=nn.Linear(conditional_dim,self.beta.shape[-1])
        torch.nn.init.constant_(self.trans_gamma.weight,val=0)
        torch.nn.init.constant_(self.trans_gamma.bias,val=0)
        torch.nn.init.constant_(self.trans_beta.weight,val=0)
        torch.nn.init.constant_(self.trans_beta.bias,val=0)
        self.eps=eps
    def forward(self,X,condition):
        mean=X.mean(-1,keepdim=True)
        std=X.std(-1,keepdim=True)
        cond_gamma=self.trans_gamma(condition)
        cond_beta=self.trans_beta(condition)
        if condition.dim()<X.dim(): #condition是固定维度
            return (self.gamma+cond_gamma).unsqueeze(1)*(X-mean)/(std+self.eps)+(self.beta+cond_beta).unsqueeze(1)
        else:#condition是sequence
            return (self.gamma+cond_gamma)*(X-mean)/(std+self.eps)+(self.beta+cond_beta)
class ESIM(nn.Module):

    def __init__(self,config,
                 dropout=0,
                 device="cpu"):
        super(ESIM, self).__init__()
        embedding_dim=config.hidden_size
        hidden_size=config.hidden_size
        self.embedding_dim=embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)


#         self._encoding=TransEncoder(config)
        self._encoding_h = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)
        self.cond_h=Conditional_LayerNorm(self.hidden_size*2,50)
        self.cond_p=Conditional_LayerNorm(self.hidden_size*2,50)
#         self._encoding_p = Seq2SeqEncoder(nn.LSTM,
#                                         self.embedding_dim,
#                                         self.hidden_size,
#                                         bidirectional=True)

        self._attention =DotProductAttention()

        self._projection = nn.Sequential(nn.Linear(2*4*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())
        self.time_embeddings = nn.Embedding(config.max_position_embeddings, 32)
        self.click_times_embeddings = nn.Embedding(32, 32)
        self.ad_age_embeddings = nn.Embedding(24, 32)
        self.ad_gender_embeddings = nn.Embedding(24, 32)
        self.advertiser_age_embeddings = nn.Embedding(24, 32)
        self.advertiser_gender_embeddings = nn.Embedding(24, 32)

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size*2,
                                           self.hidden_size,
                                           bidirectional=True)
        self.ln_create=LayerNorm(50)
        self.ln_ad=LayerNorm(50)
        self.ln_product_id=LayerNorm(50)
        self.ln_product_cat=LayerNorm(50)
        self.ln_advertiser=LayerNorm(50)
        self.ln_industry=LayerNorm(50)
        
        
        self.ln_advertiser_id_industry=LayerNorm(50)
        self.ln_product_cat_industry=LayerNorm(50)
        self.ln_product_cat_advertiser=LayerNorm(50)
        self.ln_product_id_advertiser=LayerNorm(50)
        self.ln_product_id_industry=LayerNorm(50)
        self.ln_product_id_product_cat=LayerNorm(50)


        
        self.ln_time=LayerNorm(32)
        self.ln_click_times=LayerNorm(32)
        self.ln_ad_age=LayerNorm(32)
        self.ln_ad_gender=LayerNorm(32)
        self.ln_advertiser_age=LayerNorm(32)
        self.ln_advertiser_gender=LayerNorm(32)
        
        
        self.ln_target_enc=LayerNorm(12)
        self.ln_tfidf=LayerNorm(55)
        self.ln_graph=LayerNorm(42)
        self.ln_cal=LayerNorm(31)
        self.ln_one_hot_target=LayerNorm(72)
        self.ln_tfidf_sequence=LayerNorm(6)
        self.ln_tr_sequence=LayerNorm(6)
        self.decoder_gender=nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size+55,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       2))
        self.decoder_age=nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size+55,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       10))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)
    def make_mask(self,X,valid_len):
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def replace_masked(self,tensor, mask, value):
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def forward(self,times,click_times,create_embeddings,ad_embeddings,product_id_embeddings,                product_cat_embeddings,advertiser_embeddings,industry_embeddings,target_encodings,                product_cat_industry_embeddings,product_cat_advertisers,product_id_advertiser_embeddings,                product_id_industry_embeddings,product_id_product_cats,                advertiser_id_industrys,target_embed_info,                length,cal,tfidf,tfidf_sequence,one_hot_target,tr_sequence,graph_sequence,                age,gender,encoder_hidden_states=None,encoder_extended_attention_mask=None):
        norm_time=self.ln_time(self.time_embeddings(times))
        norm_click_time=self.click_times_embeddings(click_times)
        norm_target=self.ln_one_hot_target(one_hot_target)
        premises=torch.cat([self.ln_create(create_embeddings[:,:,:]),                            self.ln_ad(ad_embeddings[:,:,:]),                            self.ln_product_id(product_id_embeddings),                            norm_target,norm_time,                            norm_click_time],dim=-1)
        hypotheses=torch.cat([self.ln_advertiser(advertiser_embeddings[:,:,:]),                            self.ln_product_cat_industry(product_cat_industry_embeddings[:,:,:]),                            self.ln_product_id_advertiser(product_id_advertiser_embeddings[:,:,:]),
                            norm_target,norm_time,\
                            norm_click_time],dim=-1)
        premises_lengths=length
        hypotheses_lengths=premises_lengths
        premises_mask = self.make_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = self.make_mask(hypotheses, hypotheses_lengths).to(self.device)

        embedded_premises = premises
        embedded_hypotheses = hypotheses

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding_h(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding_h(embedded_hypotheses,
                                            hypotheses_lengths)
        attended_premises, attended_hypotheses =            self._attention(encoded_premises,encoded_hypotheses,
                            encoded_hypotheses, hypotheses_lengths),\
            self._attention(encoded_hypotheses,encoded_premises,
                    encoded_premises,premises_lengths)
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(torch.cat([projected_premises,premises],dim=-1), premises_lengths)
        v_bj = self._composition(torch.cat([projected_hypotheses,hypotheses],dim=-1), hypotheses_lengths)
#         v_ai=self.cond_p(v_ai,self.ln_product_cat(product_cat_embeddings))
#         v_bj=self.cond_h(v_bj,self.ln_industry(industry_embeddings))
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = self.replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = self.replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        hidden = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        output_age=self.decoder_age(torch.cat([hidden,self.ln_tfidf(tfidf)],dim=-1))
        output_gender=self.decoder_gender(torch.cat([hidden, self.ln_tfidf(tfidf)],dim=-1))
        loss_age=LabelSmoothingCrossEntropy(0.1,weights=age_nums.to(self.device))
        loss_gender=LabelSmoothingCrossEntropy(0.1,weights=gender_nums.to(self.device))
        l=0.5*loss_age(output_age,age.long())+0.5*loss_gender(output_gender,gender.long())
        return l,output_age,output_gender
class ESIM_Single(nn.Module):

    def __init__(self,config,
                 dropout=0,
                 device="cpu"):
        super(ESIM_Single, self).__init__()
        embedding_dim=config.hidden_size
        hidden_size=config.hidden_size
        self.embedding_dim=embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)


#         self._encoding=TransEncoder(config)
        self._encoding_h = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)
#         self._encoding_p = Seq2SeqEncoder(nn.LSTM,
#                                         self.embedding_dim,
#                                         self.hidden_size,
#                                         bidirectional=True)

        self._attention =DotProductAttention()

        self._projection = nn.Sequential(nn.Linear(2*4*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())
        self.time_embeddings = nn.Embedding(config.max_position_embeddings, 32)
        self.click_times_embeddings = nn.Embedding(32, 32)
        self.ad_age_embeddings = nn.Embedding(24, 32)
        self.ad_gender_embeddings = nn.Embedding(24, 32)
        self.advertiser_age_embeddings = nn.Embedding(24, 32)
        self.advertiser_gender_embeddings = nn.Embedding(24, 32)

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size*2,
                                           self.hidden_size,
                                           bidirectional=True)
        self.ln_create=LayerNorm(50)
        self.ln_ad=LayerNorm(50)
        self.ln_product_id=LayerNorm(50)
        self.ln_product_cat=LayerNorm(50)
        self.ln_advertiser=LayerNorm(50)
        self.ln_industry=LayerNorm(50)
        
        
        self.ln_advertiser_id_industry=LayerNorm(50)
        self.ln_product_cat_industry=LayerNorm(50)
        self.ln_product_cat_advertiser=LayerNorm(50)
        self.ln_product_id_advertiser=LayerNorm(50)
        self.ln_product_id_industry=LayerNorm(50)
        self.ln_product_id_product_cat=LayerNorm(50)

        
        self.ln_time=LayerNorm(32)
        self.ln_click_times=LayerNorm(32)
        self.ln_ad_age=LayerNorm(32)
        self.ln_ad_gender=LayerNorm(32)
        self.ln_advertiser_age=LayerNorm(32)
        self.ln_advertiser_gender=LayerNorm(32)
        
        
        self.ln_target_enc=LayerNorm(12)
        self.ln_tfidf=LayerNorm(55)
        self.ln_graph=LayerNorm(42)
        self.ln_cal=LayerNorm(31)
        self.ln_one_hot_target=LayerNorm(72)
        self.ln_tfidf_sequence=LayerNorm(6)
        self.ln_tr_sequence=LayerNorm(6)
        self.decoder_gender=nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*2*self.hidden_size+55,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       2))
        self.decoder_age=nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*2*self.hidden_size+55,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       10))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)
    def make_mask(self,X,valid_len):
            shape=X.shape
            if valid_len.dim()==1:
                valid_len=valid_len.view(-1,1).repeat(1,shape[1])
            mask=(torch.arange(0,X.shape[1]).repeat(X.shape[0],1).to(X.device)<valid_len).float()
            return mask
    def replace_masked(self,tensor, mask, value):
        mask = mask.unsqueeze(1).transpose(2, 1)
        reverse_mask = 1.0 - mask
        values_to_add = value * reverse_mask
        return tensor * mask + values_to_add
    def forward(self,times,click_times,create_embeddings,ad_embeddings,product_id_embeddings,                product_cat_embeddings,advertiser_embeddings,industry_embeddings,target_encodings,                product_cat_industry_embeddings,product_cat_advertisers,product_id_advertiser_embeddings,                product_id_industry_embeddings,product_id_product_cats,                advertiser_id_industrys,target_embed_info,                length,cal,tfidf,tfidf_sequence,one_hot_target,tr_sequence,graph_sequence,                age,gender,encoder_hidden_states=None,encoder_extended_attention_mask=None):
#                                                          self.ln_product_id_product_cat(product_id_product_cats),
#                              self.ln_product_cat_advertiser(product_cat_advertisers),\
#                              self.ln_advertiser_id_industry(advertiser_id_industrys),\
#                              self.ln_graph(graph_sequence)
#                               self.ln_advertiser_age(self.ad_age_embeddings(target_embed_info[8])),\
#                               self.ln_advertiser_gender(self.ad_gender_embeddings(target_embed_info[9]))
#                            self.ln_product_id(product_id_embeddings[:,:,:]),\
        norm_time=self.ln_time(self.time_embeddings(times))
        norm_click_time=self.click_times_embeddings(click_times)
        norm_target=self.ln_one_hot_target(one_hot_target)
        premises=torch.cat([self.ln_create(create_embeddings[:,:,:]),                            self.ln_ad(ad_embeddings[:,:,:]),                            self.ln_advertiser(advertiser_embeddings[:,:,:]),                            self.ln_product_cat_industry(product_cat_industry_embeddings[:,:,:]),                            self.ln_product_id_advertiser(product_id_advertiser_embeddings[:,:,:]),
                            norm_target,norm_time,\
                            norm_click_time],dim=-1)
        premises_lengths=length
        premises_mask = self.make_mask(premises, premises_lengths).to(self.device)

        embedded_premises = premises

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)

        encoded_premises = self._encoding_h(embedded_premises,
                                          premises_lengths)
        attended_premises =self._attention(encoded_premises,encoded_premises,
                            encoded_premises, premises_lengths)
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)

        projected_premises = self._projection(enhanced_premises)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)

        v_ai = self._composition(torch.cat([projected_premises,premises],dim=-1), premises_lengths)
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)

        v_a_max, _ = self.replace_masked(v_ai, premises_mask, -1e7).max(dim=1)

        hidden = torch.cat([v_a_avg, v_a_max], dim=1)

        output_age=self.decoder_age(torch.cat([hidden,self.ln_tfidf(tfidf)],dim=-1))
        output_gender=self.decoder_gender(torch.cat([hidden, self.ln_tfidf(tfidf)],dim=-1))
        loss_age=LabelSmoothingCrossEntropy(0.1,weights=age_nums.to(self.device))
        loss_gender=LabelSmoothingCrossEntropy(0.1,weights=gender_nums.to(self.device))
        l=0.5*loss_age(output_age,age.long())+0.5*loss_gender(output_gender,gender.long())
        return l,output_age,output_gender


# In[14]:


def train(args, train_dataset,val_dataset,temp_dataset, model,num_workers=0):
    train_dataloader=Data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,                                     collate_fn=collate_fn,num_workers=num_workers)
    model.train()
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay =["bias", "LayerNorm.weight", "gamma","beta"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     opt = Lookahead(optimizer, k=5, alpha=0.5)
#     opt = SWA(optimizer, swa_start=10, swa_freq=5)
#     def lr_lambda(current_step):
#         return max(
#             0.0, float(115240 - (current_step+11524*2)) / float(max(1, 115240 - 11524))
#         )

#     scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")):
        logger.info("  loading optimizer and scheduler...")
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    else:
        logger.info("  No optimizer and scheduler,we build a new one")        
#     scheduler.step()
#     print(optimizer.param_groups[0]['lr'])
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=args.card_list)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
#     logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tqdm(range(
        epochs_trained, int(args.num_train_epochs)), desc="Epoch")
    for _ in train_iterator:
        start=time.time()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                if  (step + 1) % args.gradient_accumulation_steps == 0: 
                        steps_trained_in_current_epoch -= 1
                continue

            model.train()

            batch = tuple(t.to(args.device) if not isinstance(t,list) else [e.to(args.device) for e in t]for t in batch[:])
            inputs = { "length":batch[0],"times":batch[1],"click_times":batch[2],"create_embeddings": batch[3],                      "ad_embeddings": batch[4],"product_id_embeddings": batch[5],"product_cat_embeddings":batch[6],                      "advertiser_embeddings":batch[7],"industry_embeddings":batch[8],"target_encodings":batch[9],                      "product_cat_industry_embeddings":batch[10],"product_id_advertiser_embeddings":batch[11],                      "product_id_industry_embeddings":batch[12],"cal":batch[13],"tfidf":batch[14],                      "tfidf_sequence":batch[15],"one_hot_target":batch[16],"tr_sequence":batch[17],"graph_sequence":batch[18],                       "advertiser_id_industrys":batch[19],"product_cat_advertisers":batch[20],"product_id_product_cats":batch[21],                      "target_embed_info":batch[22],"age":batch[-2],"gender":batch[-1]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
#             logger.info("  step:%d loss %.3f", step,loss.item())

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
#                 opt.step()
                scheduler.step()  # Update learning rate schedule
#                 print(optimizer.param_groups[0]['lr'])
                model.zero_grad()
                global_step += 1


        if True:
#             if _==int(args.num_train_epochs)-1:
#                 opt.swap_swa_sgd()
            # Save model checkpoint
#             evaluate(args, train_dataset,model,num_workers, prefix="train")
            if _==int(args.num_train_epochs)-1:
                results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids =evaluate(args, val_dataset,model,num_workers,cross=True)
            else:
                results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids =evaluate(args, temp_dataset,model,num_workers,cross=True)
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(),os.path.join(output_dir,"model.pt"))
            if _==int(args.num_train_epochs)-1:
                np.save(os.path.join(output_dir,"age_preds.npy"),age_preds)
                np.save(os.path.join(output_dir,"gender_preds.npy"),gender_preds)
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#             torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            if args.fp16:
                torch.save(amp.state_dict(),os.path.join(output_dir, "amp.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        print(time.time()-start)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break



    return global_step, tr_loss / global_step
def evaluate(args, eval_dataset,model,num_workers=0, prefix="",cross=False):
    eval_output_dir = args.output_dir 

    results = {}

    if not os.path.exists(eval_output_dir) :
        os.makedirs(eval_output_dir)
    eval_dataloader=Data.DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=num_workers)

    # multi-gpu eval
#         if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
#             model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    age_preds = None
    age_out_label_ids = None
    gender_preds = None
    gender_out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) if not isinstance(t,list) else [e.to(args.device) for e in t]for t in batch[:])



        with torch.no_grad():
            inputs = { "length":batch[0],"times":batch[1],"click_times":batch[2],"create_embeddings": batch[3],                      "ad_embeddings": batch[4],"product_id_embeddings": batch[5],"product_cat_embeddings":batch[6],                      "advertiser_embeddings":batch[7],"industry_embeddings":batch[8],"target_encodings":batch[9],                      "product_cat_industry_embeddings":batch[10],"product_id_advertiser_embeddings":batch[11],                      "product_id_industry_embeddings":batch[12],"cal":batch[13],"tfidf":batch[14],                      "tfidf_sequence":batch[15],"one_hot_target":batch[16],"tr_sequence":batch[17],"graph_sequence":batch[18],                       "advertiser_id_industrys":batch[19],"product_cat_advertisers":batch[20],"product_id_product_cats":batch[21],                      "target_embed_info":batch[22],"age":batch[-2],"gender":batch[-1]}

            outputs = model(**inputs)
            tmp_eval_loss, age_logits,gender_logits = outputs[:3]
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps+=1
        if age_preds is None:
            age_preds = torch.softmax(age_logits,dim=-1).detach().cpu().numpy()
            age_out_label_ids = inputs["age"].detach().cpu().numpy()
        else:
            age_preds = np.append(age_preds, torch.softmax(age_logits,dim=-1).detach().cpu().numpy(), axis=0)
            age_out_label_ids = np.append(age_out_label_ids, inputs["age"].detach().cpu().numpy(), axis=0)
        if gender_preds is None:
            gender_preds = torch.softmax(gender_logits,dim=-1).detach().cpu().numpy()
            gender_out_label_ids = inputs["gender"].detach().cpu().numpy()
        else:
            gender_preds = np.append(gender_preds, torch.softmax(gender_logits,dim=-1).detach().cpu().numpy(), axis=0)
            gender_out_label_ids = np.append(gender_out_label_ids, inputs["gender"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        if not cross:
            gender_preds = (gender_preds.argmax(axis=-1)).astype(np.int8)
            age_preds = (age_preds.argmax(axis=-1)).astype(np.int8)
        gender_out_label_ids=gender_out_label_ids.astype(np.int8)
        age_out_label_ids=age_out_label_ids.astype(np.int8)
    if not cross:
        result = {"acc_age":(age_preds==age_out_label_ids).sum()/age_preds.shape[0],              "acc_gender":(gender_preds==gender_out_label_ids).sum()/gender_preds.shape[0]}
    else:
        temp_gender_preds = (gender_preds.argmax(axis=-1)).astype(np.int8)
        temp_age_preds = (age_preds.argmax(axis=-1)).astype(np.int8)
        result = {"acc_age":(temp_age_preds==age_out_label_ids).sum()/temp_age_preds.shape[0],              "acc_gender":(temp_gender_preds==gender_out_label_ids).sum()/temp_gender_preds.shape[0]}
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids


# ## train

# In[15]:


n_folds=5
kf = KFold(n_splits=n_folds, shuffle=False,random_state=20)
train_idxs=[]
val_idxs=[]
for i,(train_idx,val_idx) in enumerate(kf.split(train_features)):
    train_idxs.append(train_idx)
    val_idxs.append(val_idx)


# In[18]:


pattern='train'
# config=BertConfig(**json.load(open("./ESIM_config.json","r")),              output_hidden_states=True, output_attentions=True)
# config.hidden_size=config.hidden_size-250
config=BertConfig(**json.load(open("./ESIM_single.json","r")),              output_hidden_states=True, output_attentions=True)
config.hidden_size=config.hidden_size-550
device=torch.device("cuda:0")
age_preds_list=[]
gender_preds_list=[]

for fold in tqdm(range(5)):  
    print("第",fold)
    train_idx=train_idxs[fold][:]
    val_idx=val_idxs[fold][:]
    train_dataset=UsrDataset(train_features.iloc[train_idx])
    eval_dataset=UsrDataset(train_features.iloc[val_idx])
    temp_dataset=UsrDataset(train_features.iloc[val_idx[:50000]])
#     learning_rate =0.005
    learning_rate =0.003
    weight_decay = 0.1
    epochs =10
    batch_size = 256
    adam_epsilon=1e-8
#     model=ESIM(config,0,device=device)
    model=ESIM_Single(config,0,device=device)
    output_dir="../../model/fjw/model_ESIM_Single_wv50_0.003_shuffle_length100_reverse_"+str(fold+1)+"/"
#     output_dir="../../model/fjw/model_ESIM_Single_glove50_0.003_shuffle_length88_reverse_"+str(fold+1)+"/"
    card_list=[0,2,3]

    args=ARG(train_batch_size=batch_size,eval_batch_size=batch_size,weight_decay=weight_decay,learning_rate=learning_rate,
             adam_epsilon=adam_epsilon,num_train_epochs=epochs,warmup_steps=int(len(train_dataset)//(batch_size))*1,gradient_accumulation_steps=1,save_steps=int(len(train_dataset)//(batch_size)),
             max_grad_norm=10.0,model_name_or_path=output_dir,output_dir=output_dir,seed=42,device=device,n_gpu=-1,
            max_steps=0,output_mode="classification",fp16=False,fp16_opt_level='O1',card_list=card_list)
    if pattern=='train':
#         model.load_state_dict(torch.load(os.path.join(output_dir,"checkpoint-23048","model.pt")))
        model=model.to(args.device)
        train(args,train_dataset,eval_dataset,temp_dataset,model,4)
    else:
        model.load_state_dict(torch.load(os.path.join(output_dir,"checkpoint-93750","model.pt")))
        model=model.to(args.device)
        results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids=        evaluate(args,temp_dataset,model,8,cross=True)
        age_preds_list.append(age_preds)
        gender_preds_list.append(gender_preds)


# ## test&val

# ### test

# In[20]:


train_features=pk.load(open("../../var/fjw/test_mid/new_feature_simple.pk","rb"))
age_preds_list=[]
gender_preds_list=[]
# config=BertConfig(**json.load(open("./ESIM_config.json","r")),\
#               output_hidden_states=True, output_attentions=True)
# config.hidden_size=config.hidden_size-250
config=BertConfig(**json.load(open("./ESIM_single.json","r")),              output_hidden_states=True, output_attentions=True)
config.hidden_size=config.hidden_size-550
device=torch.device("cuda:0")
# model=ESIM(config,0,device=device)
model=ESIM_Single(config,0,device=device)
for fold in range(5):
    print("第",fold)
    test_dataset=UsrDataset(train_features)
    learning_rate =0.003
    weight_decay = 0.1
    epochs =10
    batch_size = 512
    adam_epsilon=1e-8
    output_dir="../../model/fjw/model_ESIM_Single_wv50_0.003_shuffle_length100_reverse_"+str(fold+1)+"/"
#     output_dir="../../model/fjw/model_ESIM_Single_glove50_0.003_shuffle_length88_reverse_"+str(fold+1)+"/"
    card_list=[0,2,3]
    args=ARG(train_batch_size=batch_size,eval_batch_size=batch_size,weight_decay=weight_decay,learning_rate=learning_rate,
             adam_epsilon=adam_epsilon,num_train_epochs=epochs,warmup_steps=int(len(test_dataset)//(batch_size))*1,gradient_accumulation_steps=1,save_steps=int(len(test_dataset)//(batch_size)),
             max_grad_norm=5.0,model_name_or_path=output_dir,output_dir=output_dir,seed=42,device=device,n_gpu=-1,
            max_steps=0,output_mode="classification",fp16=False,fp16_opt_level='O1',card_list=card_list)

    model.load_state_dict(torch.load(os.path.join(output_dir,"checkpoint-93750","model.pt")))
    model=model.to(args.device)
    results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids=    evaluate(args,test_dataset,model,6,cross=True)
    age_preds_list.append(age_preds)
    gender_preds_list.append(gender_preds)
age_preds_list=np.stack(age_preds_list,axis=-1).mean(axis=-1)
gender_preds_list=np.stack(gender_preds_list,axis=-1).mean(axis=-1)
test_age_preds=age_preds_list
test_gender_preds=gender_preds_list


# ### val

# In[21]:


train_features=pk.load(open("../../var/fjw/train_mid/new_feature_simple.pk","rb"))
train_age_preds=np.zeros((train_features.shape[0],10))
train_gender_preds=np.zeros((train_features.shape[0],2))
# config=BertConfig(**json.load(open("./ESIM_config.json","r")),\
#               output_hidden_states=True, output_attentions=True)
# config.hidden_size=config.hidden_size-250
config=BertConfig(**json.load(open("./ESIM_single.json","r")),              output_hidden_states=True, output_attentions=True)
config.hidden_size=config.hidden_size-550
device=torch.device("cuda:0")
# model=ESIM(config,0,device=device)
model=ESIM_Single(config,0,device=device)
for fold in range(5):
    print("第",fold)
    train_idx=train_idxs[fold][:]
    val_idx=val_idxs[fold][:]
    val_dataset=UsrDataset(train_features.iloc[val_idx])
    learning_rate =0.003
    weight_decay = 0.1
    epochs =10
    batch_size = 512
    adam_epsilon=1e-8
    output_dir="../../model/fjw/model_ESIM_Single_wv50_0.003_shuffle_length100_reverse_"+str(fold+1)+"/"
#     output_dir="../../model/fjw/model_ESIM_Single_glove50_0.003_shuffle_length88_reverse_"+str(fold+1)+"/"
    card_list=[0,2,3]
    args=ARG(train_batch_size=batch_size,eval_batch_size=batch_size,weight_decay=weight_decay,learning_rate=learning_rate,
             adam_epsilon=adam_epsilon,num_train_epochs=epochs,warmup_steps=int(len(test_dataset)//(batch_size))*1,gradient_accumulation_steps=1,save_steps=int(len(test_dataset)//(batch_size)),
             max_grad_norm=5.0,model_name_or_path=output_dir,output_dir=output_dir,seed=42,device=device,n_gpu=-1,
            max_steps=0,output_mode="classification",fp16=False,fp16_opt_level='O1',card_list=card_list)

    model.load_state_dict(torch.load(os.path.join(output_dir,"checkpoint-93750","model.pt")))
    model=model.to(args.device)
    results,age_preds,age_out_label_ids,gender_preds,gender_out_label_ids=    evaluate(args,val_dataset,model,4,cross=True)
    train_age_preds[val_idx]=age_preds[:train_age_preds[val_idx].shape[0]]
    train_gender_preds[val_idx]=gender_preds[:train_gender_preds[val_idx].shape[0]]


# In[22]:


train_age_preds.shape,train_gender_preds.shape,test_age_preds.shape,test_gender_preds.shape


# In[23]:


outputs_dict={"train_gender":train_gender_preds,"train_age":train_age_preds,"test_gender":test_gender_preds,"test_age":test_age_preds}
pk.dump(outputs_dict,open("../../var/model_ret_dicts/fjw/"+output_dir.strip().split("/")[-2][:-2]+".pk","wb"))

