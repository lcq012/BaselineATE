import random
import numpy as np
import os
import fitlog
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from utils import AspectProcessor, convert_examples_to_features, load_and_cache_examples, get_logger
from transformers import BertTokenizer
from model import BertForTokenClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from tqdm import tqdm
import pdb
import codecs
logger = get_logger("log/train.log")
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


# 设置参数
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def eval_conlleval(args, examples, tokenizer, result, convall_file, eval):
    # eg: 0:'O', 1:'B-AS', 2:'I-AS'
    id2label = {index: label for label, index in args.label2id.items()}
    def test_result_to_pair(writer):
        for example, prediction in zip(examples, result):
            line = ''
            line_token = example.text_a.split()
            label_token = example.label_a.split()
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                logger.info(example.text_a)
                logger.info(example.label_a)
                logger.info("len of tokens is not equal to len of label!")
                break
            
            step = 1
            for index in range(len_seq):
                if index >= args.max_seq_length - 2:
                    break
                cur_token = line_token[index]
                cur_label = label_token[index]
                sub_token = tokenizer.tokenize(cur_token)
                try:
                    if len(sub_token) == 0:
                        raise ValueError
                    elif len(sub_token) == 1:
                        line += cur_token + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                        step += 1
                    elif len(sub_token) > 1:
                        if cur_label.startswith("B-"):
                            line += sub_token[0] + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1
                            cur_label = "I-" + cur_label[2:]
                            sub_token = sub_token[1:]
                        for t in sub_token:
                            line += t + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1
                except Exception as e:
                    logger.warning(e)
                    logger.warning(example.text_a)
                    logger.warning(example.label_a)
                    line = ''
                    raise
            writer.write(line + '\n')
    with codecs.open(convall_file, "w", encoding="utf-8") as writer:
        test_result_to_pair(writer)
    from conlleval import return_report
    eval_result, p, r, f = return_report(convall_file)
    logger.info(''.join(eval_result))
    try:
        file_name = args.datasets + "-" + str(args.seed) + ".txt"
        with open(os.path.join(args.data_dir, file_name), "a+", encoding="utf8") as report:
            report.write(''.join(eval_result))
            report.write(eval + '\n')
            report.write("#" * 80 + "\n")
    except:
        raise
    return p, r, f



def train(args, train_dataset, dev_dataset, model, tokenizer, output_model_file, test_dataset=None):
    # pad_token是用于补全的字符
    # RandomSampler：随机采样器，返回随机采样的值，接受一个数据集做参数
    train_sampler = RandomSampler(train_dataset)
    # Dataset定义了整个数据集，Sampler提供了取数据的机制，最后由Dataloader取完成取数据的任务
    # DataLoader：
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    num_train_optimization_steps = len(train_dataloader) * args.epochs

    bert_param = [p for p in list(model.named_parameters()) if 'bert.' in p[0]]
    param_optimizer = [p for p in list(model.named_parameters()) if 'bert.' not in p[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': args.lr * 10, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': args.lr * 10, 'weight_decay': 0.0},
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    # optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    else:
        amp = None
    
    model.zero_grad()
    model.train()
    set_seed(args)
    global_step = 0
    best_f = 0.0
    for epoch in range(int(args.epochs)):
        for step, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, segments, labels, lens = batch
            output = model(inputs, masks, segments, labels=labels)
            loss = output["loss"]
            # loss = (loss-0.002).abs() + 0.002
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() 
            
            optimizer.step() # theta = theta - lr * grd
            scheduler.step() # 改变学习率的值
            model.zero_grad() # 清空梯度

            global_step += 1

            if global_step % args.eval_steps == 0:
                fitlog.add_loss(loss.item(), name="Loss", step=global_step)
                model.eval()
                p, r, f = evaluate(args, model, tokenizer, dev_dataloader, eval="dev")
                fitlog.add_metric({"Dev": {"F": f, "P": p, "R": r, "epoch": epoch}}, step=global_step)
                if epoch > 2 and f >= best_f:
                    fitlog.add_best_metric({"Dev": {"F": f, "P": p, "R": r, "epoch": epoch, "step": global_step}})
                    torch.save(model.state_dict(), output_model_file)
                    if test_dataset is not None:
                        test_p, test_r, test_f = evaluate(args, model, tokenizer, test_dataloader, eval="test")
                        fitlog.add_best_metric({"Test": {"F": test_f, "P": test_p, "R": test_r, "epoch": epoch, "step": global_step}})
                    best_f = f
                model.train()
        

def evaluate(args, model, tokenizer, eval_dataloader, eval):
    model.eval()
    out_preds, out_label_ids, out_lens = [], [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, segments, labels, lens = batch
            output = model(inputs, masks, segments, labels=labels)
            predicts = output["logits"]
            predicts = torch.argmax(predicts, dim=-1)
            out_preds.append(predicts.detach().cpu().numpy())
        torch.cuda.empty_cache()

    test_result = []
    for numpy_result in out_preds:
        test_result.extend(numpy_result.tolist())
    if eval == "test":
        p, r, f = eval_conlleval(args, args.test_examples, tokenizer, test_result, os.path.join(args.output_dir, args.eval_test_file_name), eval=eval)
    else:
        p, r, f = eval_conlleval(args, args.eval_examples, tokenizer, test_result, os.path.join(args.output_dir, args.eval_test_file_name), eval=eval)
    return p, r, f

def main():
    from config import args
    # args保留着指令行输入的参数，比如：data_dir,test_dir等等
    fitlog.set_log_dir("logs/")
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    set_seed(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # for what
    processor = AspectProcessor()
    # pdb.set_trace()
    args.label2id = processor.get_labels()
    logger.info("LABEL : {}".format(args.label2id))
    args.num_labels = len(args.label2id)
    # tokenizer 分词器用于分割一句话中的各个英文单词 
    tokenizer = BertTokenizer.from_pretrained(args.plm)
    # 将tokens转化为id
    args.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    model = BertForTokenClassification.from_pretrained(args.plm, num_labels=args.num_labels)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # if args.do_train:
    train_dataset, dev_dataset = load_and_cache_examples(args, tokenizer, processor, "train")
    test_dataset = load_and_cache_examples(args, tokenizer, processor, "test")
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    print('-------------------------------------------------Training Time-------------------------------------------------')
    train(args, train_dataset, dev_dataset, model, tokenizer, output_model_file, test_dataset)
    print('-------------------------------------------------Training Over-------------------------------------------------')
    # if args.do_eval:
    #     evaluate(args, model, tokenzier, dataname="test")
    model_ckpt = torch.load(os.path.join(args.output_dir, WEIGHTS_NAME))
    model.load_state_dict(model_ckpt)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    print('-------------------------------------------------Testing Time-------------------------------------------------')
    evaluate(args, model, tokenizer, test_dataloader, eval="test")
    fitlog.finish()
    print('-------------------------------------------------Testing Over-------------------------------------------------')

if __name__ == "__main__":
    main()