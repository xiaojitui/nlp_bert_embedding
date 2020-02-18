# Bert Model

Please download the pre-trained BERT model for English and save the model in './checkpoint/en' folder. 

If want to do Chinese NLP, please also download the pre-trained BERT model for Chinese, and save the model in './checkpoint/ch' folder.

The folder should include:

(1) bert_config.json

(2) bert_model.ckpt.data-00000-of-00001

(3) bert_model.ckpt.index

(4) bert_model.ckpt.meta

(5) vocab.txt


Please also download three original scripts provided by Google:

(1) extract_features.py

(2) modeling.py

(3) tokenization.py



The two scripts:

(1) extract_features_1.py

(2) extract_features_2.py

are customized to extract BERT embedding. 



The 'tmp' folder is used to save the temporaty graph of BERT. 


