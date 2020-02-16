#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#/ids/datasets/glove_vectors/

#/ids/datasets/nlpdata/
# In[5]:


import numpy as np


# In[6]:


path1 = './glove_vectors/glove.6B.50d.txt'


# In[77]:


def load_embedding(path):

    words = []
    idx = 0
    word_to_idx = {}
    vectors = []
    with open(path1, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word_to_idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    emb_dim = len(vectors[0])
    # add OOV and PADDING
    words.extend(['UNK', 'PAD'])
    word_to_idx[words[-2]] = len(word_to_idx)
    word_to_idx[words[-1]] = len(word_to_idx)
    vectors = np.concatenate([vectors, np.zeros((2, emb_dim))]).astype(np.float32)
    return word_to_idx, words, vectors


# In[78]:


#word_to_idx, words, emb = load_embedding(path1)


# In[79]:


#words[-5:]


# In[80]:


#word_to_idx[words[-4]], word_to_idx[words[-3]], word_to_idx[words[-2]], word_to_idx[words[-1]]


# In[82]:


#emb[-1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


import pandas as pd

df = pd.read_csv('./data/LoughranMcDonald_MasterDictionary_2014.csv')


# In[16]:


#df.head()


# In[ ]:





# In[17]:


#len(df)


# In[ ]:





# In[19]:


def load_vocabulary():
    vocab = pd.read_pickle(VOCAB_FILE)
    masks = vocab[ vocab.is_mask].index.tolist()
    words = vocab[-vocab.is_mask].index.tolist()
    vocab = masks + words + ['OOV', 'PADDING']
    oov = len(vocab) - 2
    padding = len(vocab) - 1
    assert vocab[oov] == 'OOV'
    assert vocab[padding] == 'PADDING'
    return vocab, masks, words, oov, padding 


def load_embedding_matrix():
    emb = pd.read_pickle(GLOVE_FILE)
    emb = np.concatenate([emb, np.zeros((2, emb.shape[1]))]).astype(np.float32) # add OOV and PADDING
    return emb


def load_sentiment_embedding(words):
    ''' Initialize sentiment embedding using Loughran McDonald lexicon '''
    lexicon = pd.read_csv(SENTIMENT_FILE)
    lexicon['Sentiment'] = (lexicon.Positive > 0).astype(float) - (lexicon.Negative > 0).astype(float)
    lexicon = lexicon[lexicon.Sentiment != 0][['word', 'Sentiment']].set_index('word').Sentiment.to_dict()
    sentiment_emb = [lexicon.get(w, 0) for w in words]
    sentiment_emb = np.expand_dims(sentiment_emb, -1).astype(np.float32)
    return sentiment_emb


def pad_sequences(seqs, value, maxlen=None):
    ''' Pad and truncate sequences to max length '''
    return tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post', value=value)


# In[23]:


#len(vectors[0])


# In[43]:


#names = list(word_to_idx.keys())[:10]
#xtest = np.array([np.arange(22)])


# In[44]:


#xtest


# In[45]:


#xtest.shape


# In[52]:


#import tensorflow as tf
#tf.reset_default_graph()
#X = tf.placeholder(shape=(None, 22), dtype=tf.int64, name='inputs')
#with tf.name_scope('embedding'):
    #embedding = tf.get_variable('embedding', [len(vectors), len(vectors[0])], trainable=False)
    #X_embed = tf.nn.embedding_lookup(embedding, X) # None, doc_s, sen_s, embed_s

#with tf.Session() as sess:
    # sess.run([], feed_dict={embedding: char_emb})
   # x_val = sess.run(X_embed, feed_dict = {embedding: vectors, X: xtest})


# In[53]:


#x_val


# In[ ]:





# In[ ]:





# In[83]:


def split_sentence(txt):
    sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
    sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
    sents = list(filter(None, sents))
    return sents


# In[ ]:



# In[ ]:





# In[ ]:




