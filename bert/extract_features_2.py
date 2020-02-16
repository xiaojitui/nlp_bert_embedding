# coding=utf-8
# modified from "extract_features" in BERT offical github
#

import numpy as np
import re
import tensorflow as tf
#from tensorflow.python.estimator.model_fn import EstimatorSpec

from . import modeling, tokenization
#import modeling
#import tokenization

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        # self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids



def convert_lst_to_features(lst_str, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    for (ex_index, example) in enumerate(read_examples(lst_str)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        # if ex_index < 5:
        #     tf.logging.info("*** Example ***")
        #     tf.logging.info("unique_id: %s" % (example.unique_id))
        #     tf.logging.info("tokens: %s" % " ".join(
        #         [tokenization.printable_text(x) for x in tokens]))
        #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     tf.logging.info(
        #         "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        yield InputFeatures(
            # unique_id=example.unique_id,
            # tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples_one_sample(lst_strs): # for one sample in a batch
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1

        
        
def read_examples_batch(stack_strs, doc_len): # for whole batch 
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for lst_str in stack_strs:
        
        doc_tracker = 0
        while doc_tracker < doc_len:
            if doc_tracker >= len(lst_str):
                yield InputExample(unique_id=unique_id, text_a=None, text_b=None)
            else:
                ss = lst_str[doc_tracker]
                line = tokenization.convert_to_unicode(ss)
                if not line:
                    continue
                line = line.strip()
                text_a = None
                text_b = None
                m = re.match(r"^(.*) \|\|\| (.*)$", line)
                if m is None:
                    text_a = line
                else:
                    text_a = m.group(1)
                    text_b = m.group(2)
                yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
                
            doc_tracker += 1
            unique_id += 1  

            
            
def read_examples_try(stack_strs, doc_len): # for whole batch 
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for (ex_index, lst_str) in enumerate(stack_strs):
        #lst_str = lst_str[0]
    #for lst_str in stack_strs:
        #print(ex_index, lst_str)
        doc_tracker = 0
        while doc_tracker < doc_len:
            if doc_tracker >= len(lst_str):
                yield InputExample(unique_id=unique_id, text_a=None, text_b=None)
            else:
                ss = lst_str[doc_tracker]
                line = tokenization.convert_to_unicode(ss)
                if not line:
                    continue
                line = line.strip()
                text_a = None
                text_b = None
                m = re.match(r"^(.*) \|\|\| (.*)$", line)
                if m is None:
                    text_a = line
                else:
                    text_a = m.group(1)
                    text_b = m.group(2)
                yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
                
            doc_tracker += 1
            unique_id += 1


# work with "read_examples_batch"
def convert_batch_to_features(lst_str, doc_len, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    for (ex_index, example) in enumerate(read_examples_batch(lst_str, doc_len)):
        
        if example.text_a is None: # doc padding
            input_ids = [0] * seq_length
            input_mask = [0] * seq_length
            input_type_ids = [0] * seq_length

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            yield InputFeatures(
                # unique_id=example.unique_id,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

        else: 
            
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
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

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            # if ex_index < 5:
            #     tf.logging.info("*** Example ***")
            #     tf.logging.info("unique_id: %s" % (example.unique_id))
            #     tf.logging.info("tokens: %s" % " ".join(
            #         [tokenization.printable_text(x) for x in tokens]))
            #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     tf.logging.info(
            #         "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                # unique_id=example.unique_id,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

        
    

                    
# directly work with X_batch raw data                    
def convert_all_to_features(X_input, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    for (ex_index, example) in enumerate(X_input):
        if example.text_a is None: # doc padding
            input_ids = [0] * seq_length
            input_mask = [0] * seq_length
            input_type_ids = [0] * seq_length

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            yield InputFeatures(
                # unique_id=example.unique_id,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

        else: 
            
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
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

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            # if ex_index < 5:
            #     tf.logging.info("*** Example ***")
            #     tf.logging.info("unique_id: %s" % (example.unique_id))
            #     tf.logging.info("tokens: %s" % " ".join(
            #         [tokenization.printable_text(x) for x in tokens]))
            #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     tf.logging.info(
            #         "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                # unique_id=example.unique_id,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

        
    
