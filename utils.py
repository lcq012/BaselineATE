import torch, os
from torch.utils.data import TensorDataset
import pdb

import logging

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
    
logger = get_logger("log/train.log")

class InputExample(object):
    def __init__(self, guid, text_a, label_a, text_b=None, label_b=None):
        self.guid = guid
        self.text_a = text_a
        self.label_a = label_a
        self.text_b = text_b
        self.label_b = label_b

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_txt(cls, input_path):
        lines = []
        with open(os.path.join(input_path, 'sentence.txt'), 'r', encoding="utf8") as data_sent, \
            open(os.path.join(input_path, 'target.txt'), 'r', encoding="utf8") as data_tagt:
            logger.info("load file {}/{}".format(os.path.join(input_path, 'sentence.txt'), os.path.join(input_path, 'target.txt')))
            for sent, tagt in zip(data_sent.readlines(), data_tagt.readlines()):
                sent = sent.strip()
                tagt = tagt.strip()
                lines.append((sent, tagt))
        return lines

class AspectProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "dev")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test")), "test")
    def get_labels(self):
        return {"O": 0, "B-AS": 1, "I-AS": 2}
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            sent, tagt= line
            examples.append(InputExample(guid=guid, text_a=sent, label_a=tagt, text_b=None, label_b=None))
            # inputExamle.[text_a / label_a]
        return examples

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, tokens_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tokens_len = tokens_len

def load_and_cache_examples(args, tokenizer, processor, dataname="train"):
    # processor = AspectProcessor()
    max_seq_length = args.max_seq_length
    examples1 = None
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
        examples1 = processor.get_dev_examples(args.data_dir)
        args.eval_examples = examples1
    elif dataname == "test":
        examples = processor.get_test_examples(args.test_dir)
        args.test_examples = examples
    else:
        raise ValueError("dataname parameters error !")
    try:
        _features = convert_examples_to_features(examples, 
                                                args.label2id, 
                                                max_seq_length,
                                                tokenizer, 
                                                cls_token = tokenizer.cls_token,
                                                sep_token = tokenizer.sep_token,
                                                pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                cls_token_segment_id=2 if 'xlnet' in args.plm else 0,
                                                pad_token_segment_id=4 if 'xlnet' in args.plm else 0,
                                                pad_token_label_id=-1,
                                                output_dir=args.output_dir)
        if examples1 is not None:
            _features1 = convert_examples_to_features(examples1, 
                                                args.label2id, 
                                                max_seq_length,
                                                tokenizer, 
                                                cls_token = tokenizer.cls_token,
                                                sep_token = tokenizer.sep_token,
                                                pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                cls_token_segment_id=2 if 'xlnet' in args.plm else 0,
                                                pad_token_segment_id=4 if 'xlnet' in args.plm else 0,
                                                pad_token_label_id=-1,
                                                output_dir=args.output_dir)
    except:
        print(dataname)
        raise
    
    def convert_feature(features):
        all_inputs = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.tokens_len for f in features], dtype=torch.long)
        # all_poss = torch.tensor([f.pos for f in features], dtype=torch.long)
        # all_graphs = torch.tensor([f.graph for f in features], dtype=torch.float)
        dataset = TensorDataset(all_inputs, all_masks, all_segments, all_labels, all_lens)
        return dataset
    
    dataset = convert_feature(_features)
    if examples1 is not None:
        dataset1 = convert_feature(_features1)
        return dataset, dataset1
    return dataset


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer,
                                            cls_token='[CLS]', sep_token='[SEP]',
                                            pad_token_id=0,
                                            sequence_a_segment_id=0,
                                            sequence_b_segment_id=1,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            pad_token_label_id=-1,
                                            mask_padding_with_zero=True,
                                            output_dir=None):
    def _reseg_token_label(tokens, labels, tokenizer):
        assert len(tokens) == len(labels)
        ret_tokens, ret_labels = [], []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token) # 有可能有多个token
            ret_labels.append(label)
            if len(sub_token) == 1: 
                continue
            if label.startswith("B") or label.startswith("I"): # hot -> h ##o ##t  b-as i-as i-as          # hot dog (b-as i-as i-as)
                sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token)-1))
            elif label.startswith("O"):
                sub_label = label
                ret_labels.extend([sub_label] * (len(sub_token)-1))
            else:
                raise ValueError
        
        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = example.text_a.split()
        labels_a = example.label_a.split()

        def inputIdMaskSegment(tmp_tokens, tmp_labels, tmp_segment_id):
            # tmp_tokens只是根据空格分割的
            tokens, labels = _reseg_token_label(tmp_tokens, tmp_labels, tokenizer)
            # tokens已经分割成tokenizer词表中存在的token了
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]

            label_ids = [label2id[label] for label in labels]
            pad_label_id = label2id["O"] if pad_token_label_id == -1 else pad_token_label_id

            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [tmp_segment_id] * len(tokens)
            label_ids = [pad_label_id] + label_ids + [pad_label_id]

            tokens_len = len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([pad_token_id] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([tmp_segment_id] * padding_length)
            label_ids += ([pad_label_id] * padding_length)
            assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids) == max_seq_length
            return input_ids, input_mask, segment_ids, label_ids, tokens_len

        input_ids, input_mask, segment_ids, label_ids, tokens_len = inputIdMaskSegment(tokens_a, labels_a, sequence_a_segment_id) # 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_a]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, 
                                      label_ids=label_ids, tokens_len=tokens_len))
    
    return features