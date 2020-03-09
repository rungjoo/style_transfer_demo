# -*- coding: utf-8 -*-
import torch
from transformers import *
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import sys
sys.path.insert(0, "/project/work/demo/visual_v4_0/")
from dis_model import *
dismodel = findattribute().cuda()
dismodel_name='cls_model_3'
dismodel.load_state_dict(torch.load('./visual_v4_0/models/{}'.format(dismodel_name)))
dismodel.eval()

sys.path.insert(0, "/project/work/demo/nobert_v3.3/")
from gen_model import *
genmodel = styletransfer().cuda()
genmodel_name='gen_model_2'
genmodel.load_state_dict(torch.load('./nobert_v3.3/models/{}'.format(genmodel_name)))
genmodel.eval()

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

import json
f = open('gpt_yelp_vocab.json')
token2num = json.load(f)

num2token = {}
for key, value in token2num.items():
    num2token[value] = key


def get_sentence(sentence, alpha=0.5, beta=0.1):
    """data setting"""
    neg_labels = [] # negative labels
    neg_labels.append([1,0])
    neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

    pos_labels = [] # positive labels
    pos_labels.append([0,1])
    pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()
                       
    """data input"""    
    token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
    ori_length = token_idx.shape[1]
    
    """sentiment classifier"""
    dis_out = dismodel.discriminator(token_idx)    
    sent_prob = F.softmax(dis_out, 1).squeeze(0)
    sent_cls = torch.argmax(sent_prob).item()
    if sent_cls == 0:
        fake_attribute = pos_attribute
        sentiment = 0
        sentiment_str = 'negative'
        sentimen_trans = 'positive'
    else: # sent_cls == 1
        fake_attribute = neg_attribute
        sentiment = 1
        sentiment_str = 'positive'
        sentimen_trans = 'negative'

    # delete model
    max_len = int(token_idx.shape[1]*(1-beta))
    del_idx = token_idx
    for k in range(max_len):
        del_idx = dismodel.att_prob(del_idx, sentiment)           
        dis_out = dismodel.discriminator(del_idx)    
        sent_prob = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
        if sent_prob < alpha:
            break     

    del_list = del_idx.squeeze(0).cpu().tolist() # list
    del_sen =''
    for x in range(len(del_list)):            
        token = num2token[del_list[x]].strip('Ġ')
        del_sen += token
        del_sen += ' '
    del_sen = del_sen.strip()

    del_percent = 100-(del_idx.shape[1])/(token_idx.shape[1]) * 100

    enc_out = genmodel.encoder(del_idx)
    gen_sen_2 = genmodel.generated_sentence(enc_out, fake_attribute, ori_length)
    
    
#     return sentiment_str, del_sen, del_percent, gen_sen_2.rstrip('<|endoftext|>')
    return gen_sen_2.rstrip('<|endoftext|>'), sentimen_trans
