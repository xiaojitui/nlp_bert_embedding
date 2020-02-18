In this folder, several scripts were tested to extract word/character embeddings. 

- 'embedding_m0.py' uses TF-Estimator to do prediction to get BERT embeddings.
- 'embedding_m1.py' gets the tensors in BERT and uses Sess.run to do prediction to get BERT embeddings. 
- 'embedding_m2.py' generates a temporary graph for BERT and loads the graph to get BERT embeddings. 


The speed: 'embedding_m2.py' is faster than 'embedding_m1.py' which is after than 'embedding_m0.py'
