#!/usr/bin/env python
# coding: utf-8

# # use TF-Estimator to do prediction  

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("../")
sys.path.append("../bert")


from bert import modeling, tokenization
from bert.extract_features import input_fn_builder, model_fn_builder, InputExample, InputFeatures
from bert.extract_features import convert_examples_to_features, convert_examples_to_features_1, convert_examples_to_features_2
#from tensorflow.python.estimator.estimator import Estimator
#from tensorflow.python.estimator.run_config import RunConfig

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

# In[3]:


def extract_v1(sentence, estimator, tokenizer, sen_len = 15):
    example = [InputExample(unique_id=0, text_a=sentence, text_b=None)]
    features = convert_examples_to_features_1(examples=example,
                                                  seq_length=sen_len,
                                                  tokenizer=tokenizer)
    input_fn = input_fn_builder(features=features,
                                                  seq_length=sen_len)

    outputs = []
    for output in estimator.predict(input_fn):
        outputs.append(output)
    
    return outputs[0]


# In[ ]:





# In[31]:


def extracts_v1(sentences, estimator, tokenizer, sen_len = 15):

    examples = []
    for idx, sentence in enumerate(sentences):
        examples.append(InputExample(unique_id=idx, text_a=sentence, text_b=None))
    features = convert_examples_to_features(examples=examples,
                                                  seq_length=sen_len,
                                                  tokenizer=tokenizer) #, get_cls = get_cls)
    
    input_fn = input_fn_builder(features=features,
                                                  seq_length=sen_len)
    outputs = []
    for output in estimator.predict(input_fn):
        outputs.append(output)

    return outputs


# In[23]:


def concat_layers(results, layers = 4):
    
    all_layers = []
    for i in range(layers-1, -1, -1):
        name = 'layer_output_' + str(i)
        all_layers.append(results[name])
        
    all_layers = np.concatenate(all_layers, axis = 1)
    
    return all_layers


# In[6]:


def extracts_pad(add_n, estimator, select_layers, sen_len = 15):

    features = []
   
    for _ in range(add_n):
        pad_feature = InputFeatures(
            unique_id=[0],
            tokens='[PAD]',
            input_ids=[0],
            input_mask=[0],
            input_type_ids=[0])
        features.append(pad_feature)
    
    input_fn = input_fn_builder(features=features,
                                                  seq_length=sen_len)
    outputs = []
    layers = len(select_layers)
    for output in estimator.predict(input_fn):
        outputs.append(concat_layers(output, layers))

    return outputs


# In[7]:


def get_token(doc, tokenizer, doc_len = 5, sen_len = 15):

    doc_token = []
    doc_mask = []
    for sen in doc:
        
        _token = tokenizer.tokenize(sen[:sen_len])
        _input_id = tokenizer.convert_tokens_to_ids(_token)
        _input_mask = [1] * len(_input_id)
        _segment_id = [0] * len(_input_id)
        
        while len(_input_id) < sen_len:
            _input_id.append(0)
            _input_mask.append(0)
            _segment_id.append(0) 
        doc_token.append(_input_id)
        doc_mask.append(_input_mask)
        
    while len(doc_token) < doc_len:
        doc_token.append([0]*sen_len)
        doc_mask.append([0]*sen_len)
    return np.array(doc_token), np.array(doc_mask)


# In[8]:


def get_seq(doc_mask):
    doc_seq_len = []
    sen_seq_len = []

    if 0 in doc_mask.T[0]:
        doc_seq_len = list(doc_mask.T[0]).index(0)
    else:
        doc_seq_len = len(doc_mask)

    for sen in doc_mask:
        if 0 in sen:
            cur_sen_seq = list(sen).index(0)
        else:
            cur_sen_seq = len(sen)
        sen_seq_len.append(cur_sen_seq)

    #doc_seq_len = np.array(doc_seq_len)
    #sen_seq_len = np.array(sen_seq_len)
    return doc_seq_len, sen_seq_len


# In[9]:


def get_batch_seq(X_batch, tokenizer, doc_len = 5, sen_len = 15):
    
    batch_doc_seq = []
    batch_sen_seq = []
    batch_doc_token = []
    for X in X_batch:
        doc_token, doc_mask = get_token(X, tokenizer, doc_len, sen_len)
        doc_seq_len, sen_seq_len = get_seq(doc_mask)
        batch_doc_seq.append(doc_seq_len)
        batch_sen_seq.extend(sen_seq_len)
        batch_doc_token.append(doc_token)
        
    return np.array(batch_doc_seq), np.array(batch_sen_seq), np.array(batch_doc_token)


# In[10]:


def get_emb(doc, estimator, tokenizer, select_layers, sen_len = 15):
    results = extracts_v1(doc, estimator, tokenizer, sen_len)
    
    emb = []
    
    layers = len(select_layers)
    for result in results:
        _emb = concat_layers(result, layers)
        emb.append(_emb)
    return np.array(emb)


# In[1]:


# add '[PAD]' token for paddings
def get_batch_emb(X_batch, estimator, tokenizer, select_layers, doc_len = 5, sen_len = 15):
    
    batch_emb = []
    for X in X_batch:
        
        if len(X) >= doc_len:
            X = X[:doc_len]
            emb = get_emb(X, estimator, tokenizer, select_layers, sen_len)
            
            
        if len(X) < doc_len:
            emb = get_emb(X, estimator, tokenizer, select_layers, sen_len)
            add_n = doc_len - len(X)
            pad_emb = extracts_pad(add_n, estimator, select_layers, sen_len)
            emb = np.concatenate((emb, pad_emb))
            
        batch_emb.append(emb)
        
    return np.array(batch_emb)


# In[2]:


# use random embedding for paddings
def get_batch_emb_1(X_batch, estimator, tokenizer, select_layers, doc_len = 5, sen_len = 15):
    
    batch_emb = []
    for X in X_batch:
        if len(X) >= doc_len:
            X = X[:doc_len]
            emb = get_emb(X, estimator, tokenizer, select_layers, sen_len)
            
            
        if len(X) < doc_len:
            emb = get_emb(X, estimator, tokenizer, select_layers, sen_len)
            add_n = doc_len - len(X)
            pad_emb = np.random.uniform(-1, 1, (add_n, sen_len, emb.shape[-1]))
            emb = np.concatenate((emb, pad_emb))
            
        batch_emb.append(emb)
        
    return np.array(batch_emb)


# In[ ]:





# In[13]:


def prepare_bert(bert_path, bert_config_file, bert_vocab_file, init_checkpoint, select_layers):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model_fn = model_fn_builder(bert_config=bert_config,
                                                  init_checkpoint=init_checkpoint,
                                                  layer_indexes=select_layers,
                                                  use_tpu=False,
                                                  use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(model_fn=model_fn,
                                            model_dir=bert_path,
                                            use_tpu=False,
                                            predict_batch_size=32,
                                            config=tf.contrib.tpu.RunConfig())
                                            #config=tf.contrib.tpu.RunConfig(master=None, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=8, per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)))

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    #estimator = Estimator(model_fn, config=RunConfig(session_config=config), params = {'batch_size': 32}, model_dir=MODEL_DIR) 


    tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=False)
    
    return estimator, tokenizer


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## test section


bert_path = '../bert/checkpoint/'
bert_config_file = '../bert/checkpoint/bert_config.json'
bert_vocab_file = '../bert/checkpoint/vocab.txt'
init_checkpoint = '../bert/checkpoint/bert_model.ckpt'
#select_layers = [-1]
select_layers = [-1, -2, -3, -4]
doc_len = 7
sen_len = 15



epochs = 4
batch_size = 2
n_iters = len(X)//batch_size
estimator, tokenizer = prepare_bert(bert_path, bert_config_file, bert_vocab_file, init_checkpoint, select_layers)


for epoch in range(epochs):
    print('epcoh: ', epoch)
    t1 = time.time()
    for n_iter in tqdm(range(n_iters), total = n_iters):
        
        idx = np.random.choice(len(X), batch_size, replace = False)
        X_batch = [X[k] for k in idx]
        
        batch_doc_seq, batch_sen_seq, batch_doc_token = get_batch_seq(X_batch, tokenizer, doc_len, sen_len)
        batch_emb = get_batch_emb_1(X_batch, estimator, tokenizer, select_layers, doc_len, sen_len)
        
        print(batch_emb.shape)

    print('time:', time.time() - t1)


# In[ ]:




