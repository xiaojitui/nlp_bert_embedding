#!/usr/bin/env python
# coding: utf-8

# # use Sess.run to do prediction  

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
sys.path.append("../bert")

import os
import collections
from bert import modeling

from tqdm import tqdm
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore Info and Warning
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()


# In[ ]:





# ## sample dataset

# In[2]:


X = [['我想去看电影'], 
     ['我想去', '看电影', '中文自然语言处理', '包括字向量'], 
     ['包括字向量', '中文自然语言处理包括字向量'], 
     ['我想', '看电', '中文语言处理', '字向量', '语言处理', '中文语言', '中'], 
     ['我想去', '中文语言处理'], 
     ['看电影', '中处理']]
y = [-0.8, -0.1, 0.22, 0.5, -0.3, 0.4]

X_train = X[:4]
y_train = y[:4]
X_test = X[4:]
y_test = y[4:]


# In[ ]:





# ## functions

# In[4]:


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    vocab.setdefault("blank",2)
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


# In[5]:


def inputs(vectors,maxlen=10):
    length=len(vectors)
    if length>=maxlen:
        return  vectors[0:maxlen],[1]*maxlen,[0]*maxlen
    else:
        input_vec =vectors+[0]*(maxlen-length)
        mask=[1]*length+[0]*(maxlen-length)
        segment=[0]*maxlen
        return input_vec,mask,segment


# In[6]:


def get_layers(ret, layer_n = 4):
    rets = []
    for ele in ret: #ele is 12*768
        cur_r = []
        for j in range(len(ele)-layer_n, len(ele)):
            cur_r.extend(ele[j])

        rets.append(cur_r)

    rets = np.stack(rets)

    return rets


# In[7]:


def prepare_bert(bert_config_file, bert_vocab_file):
    
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    
    is_training=False
    use_one_hot_embeddings=False

    input_ids_p=tf.placeholder(shape=[None,None],dtype=tf.int32,name="input_ids_p")
    input_mask_p=tf.placeholder(shape=[None,None],dtype=tf.int32,name="input_mask_p")
    segment_ids_p=tf.placeholder(shape=[None,None],dtype=tf.int32,name="segment_ids_p")

    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids_p,
            input_mask=input_mask_p,
            token_type_ids=segment_ids_p,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
    
    
    di=load_vocab(vocab_file = bert_vocab_file)
    
    #embedding = tf.squeeze(model.get_all_encoder_layers())
    embedding = tf.squeeze(model.get_all_encoder_layers())
    embedding = tf.transpose(embedding, perm = [1, 0, 2])
    
    return di, embedding


# In[ ]:





# In[8]:


## if multiple sentences, use list(text[i]) instead of list(text[j])

def get_emb(X, init_checkpoint, di, embedding, get_cls = True, select_layers=4, sen_len = 10):
    
    sess=tf.Session()
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, init_checkpoint)
    
    X_emb = []
    ids_emb = []
    mask_emb = []
    seg_emb = []
    
    for text in X:
        #vectors = [di.get("[CLS]")] + [di.get(i) if i in di else di.get("[UNK]") for i in list(text)] + [di.get("[SEP]")]
        if get_cls:
            vectors = [di.get("[CLS]")] + [di.get(i) if i in di else di.get("[UNK]") for i in list(text)] + [di.get("[SEP]")]
        else:
            vectors = [di.get(i) if i in di else di.get("[UNK]") for i in list(text)]

        input_vec, mask, segment = inputs(vectors, sen_len)

        input_ids = np.reshape(np.array(input_vec), [1, -1])
        input_mask = np.reshape(np.array(mask), [1, -1])
        segment_ids = np.reshape(np.array(segment), [1, -1])


        ret=sess.run(embedding,feed_dict={"input_ids_p:0":input_ids,"input_mask_p:0":input_mask,"segment_ids_p:0":segment_ids})
        #seq_l = ret.shape()
        #ret=ret.reshape(())

        rets = get_layers(ret, select_layers)
        
        X_emb.append(rets)
        ids_emb.append(input_ids)
        mask_emb.append(input_mask)
        seg_emb.append(segment_ids)
     
    X_emb = np.stack(X_emb)
    ids_emb = np.stack(ids_emb)
    mask_emb = np.stack(mask_emb)
    seg_emb = np.stack(seg_emb)
    #return  rets, input_ids, input_mask, segment_ids
    return X_emb, ids_emb, mask_emb, seg_emb


# In[ ]:





# In[9]:


def get_pad(init_checkpoint, di, embedding, select_layers=4, add_n = 1, sen_len = 10):
    
    sess=tf.Session()
    restore_saver = tf.train.Saver()
    restore_saver.restore(sess, init_checkpoint)
    
    pad_emb = []
    
    for _ in range(add_n):
        vectors = [di.get("[PAD]")]

        input_vec, mask, segment = inputs(vectors, sen_len)

        input_ids = np.reshape(np.array(input_vec), [1, -1])
        input_mask = np.reshape(np.array(mask), [1, -1])
        segment_ids = np.reshape(np.array(segment), [1, -1])


        ret=sess.run(embedding,feed_dict={"input_ids_p:0":input_ids,"input_mask_p:0":input_mask,"segment_ids_p:0":segment_ids})
        #seq_l = ret.shape()
        #ret=ret.reshape(())

        rets = get_layers(ret, select_layers)
        
        pad_emb.append(rets)
 
    pad_emb = np.stack(pad_emb)

    return pad_emb


# In[ ]:





# In[20]:


def get_batch_emb(X_batch, init_checkpoint, di, embedding, get_cls = True, select_layers=4, doc_len = 5, sen_len = 10):
    
    batch_emb = []
    for X in X_batch:
        
        if len(X) >= doc_len:
            
            X = X[:doc_len]
            emb, _, _, _ = get_emb(X, init_checkpoint, di, embedding, get_cls, select_layers, sen_len)
            
        if len(X) < doc_len:
            emb, _, _, _ = get_emb(X, init_checkpoint, di, embedding, get_cls, select_layers, sen_len)
            add_n = doc_len - len(X)
            pad_emb = get_pad(init_checkpoint, di, embedding, select_layers, add_n, sen_len)
            emb = np.concatenate((emb, pad_emb))
        
        
        batch_emb.append(emb)
        
    return np.array(batch_emb)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## test section

# ### single batch

# In[12]:


#bert_path = '../bert/checkpoint/'
bert_config_file = '../bert/checkpoint/bert_config.json'
bert_vocab_file = '../bert/checkpoint/vocab.txt'
init_checkpoint = '../bert/checkpoint/bert_model.ckpt'
#select_layers = [-4, -3, -2, -1]
#select_layers = [-1]

select_layers = 4
doc_len = 7
sen_len = 15

get_cls = True



# ### epochs

# In[ ]:


di, embedding = prepare_bert(bert_config_file, bert_vocab_file)


# In[ ]:

epochs = 4
batch_size = 2
n_iters = len(X)//batch_size

for epoch in range(epochs):
    print('epcoh: ', epoch)
    t1 = time.time()
    for n_iter in tqdm(range(n_iters), total = n_iters):
        
        idx = np.random.choice(len(X), batch_size, replace = False)
        X_batch = [X[k] for k in idx]
        
        X_emb = get_batch_emb(X_batch, init_checkpoint, di, embedding, get_cls, select_layers, doc_len, sen_len)
        print(X_emb.shape)

    print('time:', time.time() - t1)
# In[ ]:


