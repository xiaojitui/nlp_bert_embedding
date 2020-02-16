import os
import numpy as np
import tensorflow as tf
import bert
from bert import extract_features as _
from tqdm import tqdm

BERT_PATH = 'uncased_L-12_H-768_A-12'
MAX_SEQ_LENGTH = 512

def show_object_info(obj):
    output = ['{']
    for k in dir(obj):
        if k[0] != '_':
            output.append('  \'%s\': %s,' % (k, getattr(obj, k)))
    output.append('}')
    print('\n'.join(output))
    return

def load_config():
    return bert.modeling.BertConfig.from_json_file(
        os.path.join(BERT_PATH, 'bert_config.json')
    )
    
def load_tokenizer():
    do_lower_case = 'uncased' in BERT_PATH
    return bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(BERT_PATH, 'vocab.txt'), 
        do_lower_case=do_lower_case
    )

def create_model_fn():
    bert_config = load_config()
    layer_indexes = [-1] # list(range(bert_config.num_hidden_layers))
    return bert.extract_features.model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=os.path.join(BERT_PATH, 'bert_model.ckpt'),
        layer_indexes=layer_indexes,
        use_tpu=False,
        use_one_hot_embeddings=False
    )

def create_estimator(batch_size):
    return tf.contrib.tpu.TPUEstimator(
        model_fn=create_model_fn(),
        config=tf.contrib.tpu.RunConfig(),
        predict_batch_size=batch_size,
        use_tpu=False,
    )

def convert_example_to_features(unique_id, tokens_a, tokens_b, tokenizer):
    """Variation of bert.extract_features.convert_examples_to_features"""
    
    seq_length = MAX_SEQ_LENGTH
    
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        bert.extract_features._truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return dict(
        unique_ids=unique_id,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids
    )

def create_input_fn(df, col_a, col_b=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    INPUT_TYPES = {
        'unique_ids': tf.int32, 
        'input_ids': tf.int32,
        'input_mask': tf.int32,
        'input_type_ids': tf.int32,
    }
    INPUT_SHAPES = {
        'unique_ids': (), 
        'input_ids': (MAX_SEQ_LENGTH), 
        'input_mask': (MAX_SEQ_LENGTH), 
        'input_type_ids': (MAX_SEQ_LENGTH),
    }
    keys = list(INPUT_TYPES.keys())
    tokenizer = load_tokenizer()
    
    def input_generator():
        for i, row in tqdm(df.iterrows(), total=len(df), position=0):
            features = convert_example_to_features(i, row[col_a], row[col_b] if col_b else None, tokenizer)
            yield {k: features[k] for k in keys}
        return

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        d = tf.data.Dataset.from_generator(input_generator, INPUT_TYPES, INPUT_SHAPES)
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn

def predict(estimator, input_fn):
    bert_config = load_config()
    results = estimator.predict(input_fn, yield_single_examples=True)
    for result in results:
        yield result
#         embeddings = [
#             np.expand_dims(result[f'layer_output_{i}'], 0)
#             for i in range(bert_config.num_hidden_layers)
#         ]
#         embeddings = np.concatenate(embeddings) # shape = (num_layers, num_words, num_hidden_size)
#         yield result['unique_id'], embeddings
    return
