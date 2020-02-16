#!/usr/bin/env python
# coding: utf-8

# # use Graph to do prediction  

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
sys.path.append("../bert")

import os
from bert import modeling, tokenization
from bert.extract_features_2 import convert_all_to_features, read_examples_batch, read_examples_try

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

from tqdm import tqdm
import time 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore Info and Warning
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()




# In[ ]:





# In[ ]:





# ## sample dataset

# In[3]:


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

# In[ ]:





# In[4]:


def create_graph(graph_file, bert_config_file, init_checkpoint, max_seq_len, select_layers, output_dir = '../bert/tmp'):
    #tf.reset_default_graph()
    #from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    tf.gfile.MakeDirs(output_dir)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None,max_seq_len), 'input_mask')
    input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

    input_tensors = [input_ids, input_mask, input_type_ids]

    
    model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=False)

    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
    all_layers = []
    if len(select_layers) == 1:
        encoder_layer = model.all_encoder_layers[select_layers[0]]
    else:
        for layer in select_layers:
            all_layers.append(model.all_encoder_layers[layer])
        encoder_layer = tf.concat(all_layers, -1)


    #output_tensors = [encoder_layer]
    pooled = tf.identity(encoder_layer, 'final_encodes')
    output_tensors = [pooled]
        
    tmp_g = tf.get_default_graph().as_graph_def()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
        #[print(n.name) for n in output_tensors]
        dtypes = [n.dtype for n in input_tensors]
        #[print(n.name) for n in input_tensors]
        tmp_g = optimize_for_inference(
            tmp_g,
            [n.name[:-2] for n in input_tensors],
            [n.name[:-2] for n in output_tensors],
            [dtype.as_datatype_enum for dtype in dtypes],
            False)
    tmp_file = graph_file
    with tf.gfile.GFile(tmp_file, 'wb') as f:
        f.write(tmp_g.SerializeToString())
    return tmp_file


# In[ ]:





# In[5]:


def get_estimator(bert_config_file, init_checkpoint, max_seq_len, select_layers, graph_file = '../bert/tmp/graph'):
    #from tensorflow.python.estimator.estimator import Estimator
    #from tensorflow.python.estimator.run_config import RunConfig
    #from tensorflow.python.estimator.model_fn import EstimatorSpec

    if os.path.exists(graph_file):
        graph_path =graph_file
    else:
        graph_path = create_graph(graph_file, bert_config_file, init_checkpoint, max_seq_len, select_layers)
    
    def model_fn(features, labels, mode, params):
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        input_names = ['input_ids', 'input_mask', 'input_type_ids']

        encoder_layer = tf.import_graph_def(graph_def,
                                     input_map={k + ':0': features[k] for k in input_names},
                                     return_elements=['final_encodes:0'])
        predictions = {
            # 'client_id': client_id,
            'encodes': encoder_layer[0]
        }
            
        return EstimatorSpec(mode=mode, predictions=predictions)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
    #config.log_device_placement = False
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                     params={'batch_size': 32}, model_dir='../bert/tmp')


# In[ ]:





# In[6]:


def input_fn_builder(X_feature,  max_seq_len = 15):
    def gen():
        for i in range(1):
            tmp_f = list(X_feature)
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

    def input_fn():
        #for i in gen():
            #print(i)
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32,
                          },
            output_shapes={
                'input_ids': (None, max_seq_len),
                'input_mask': (None, max_seq_len),
                'input_type_ids': (None, max_seq_len)}).prefetch(10))

    return input_fn


# In[ ]:





# In[7]:


def get_batch_emb_2(input_fn, estimator, doc_len = 5, sen_len = 15):
    
    #for X in X_batch:
    #input_fn = input_fn_builder(X_batch, tokenizer, doc_len, sen_len)
    
    result = estimator.predict(input_fn)
    emb = []
    for rq in result:
        emb.append(rq['encodes'])
    emb = np.array(emb)

    batch_emb = np.reshape(emb, (-1, doc_len, sen_len, emb.shape[-1]))
      
    return batch_emb


# In[ ]:


def get_batch_seq(X_batch, doc_len):
    
    batch_doc_seq = []
    batch_sen_seq = []

    for X in X_batch:
        batch_doc_seq.append(len(X))
        sen_seq = [len(k) for k in X]
        if len(sen_seq) < doc_len:
            sen_seq.extend([0]*(doc_len - len(sen_seq)))
       
        batch_sen_seq.extend(sen_seq)
        
    return np.array(batch_doc_seq), np.array(batch_sen_seq)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## test section 

# In[8]:


bert_path = '../bert/checkpoint/'
bert_config_file = '../bert/checkpoint/bert_config.json'
bert_vocab_file = '../bert/checkpoint/vocab.txt'
init_checkpoint = '../bert/checkpoint/bert_model.ckpt'
#select_layers = [-1]
select_layers = [-4, -3, -2, -1] #[-1, -2, -3, -4]
doc_len = 7
sen_len = 15


# In[ ]:





# In[9]:


tokenizer = tokenization.FullTokenizer(bert_vocab_file)
estimator = get_estimator(bert_config_file, init_checkpoint, sen_len, select_layers)


# In[ ]:




epochs = 4
batch_size = 2
n_iters = len(X)//batch_size

# In[21]:

            
            
for epoch in range(epochs):
    print('epcoh: ', epoch)
    t1 = time.time()
    for n_iter in tqdm(range(n_iters), total = n_iters):
        
        idx = np.random.choice(len(X), batch_size, replace = False)
        X_batch = [X[k] for k in idx]
        a, b = get_batch_seq(X_batch, doc_len)
        X_batch_data = read_examples_batch(X_batch, doc_len)
        X_batch_feature = convert_all_to_features(X_batch_data, sen_len, tokenizer)
        X_batch_input = input_fn_builder(X_batch_feature, sen_len)
        X_emb = get_batch_emb_2(X_batch_input, estimator, doc_len, sen_len)
        print(len(a), len(b), X_emb.shape)
    
    print('time:', time.time() - t1)
# In[ ]:





# In[ ]:





# In[ ]:




