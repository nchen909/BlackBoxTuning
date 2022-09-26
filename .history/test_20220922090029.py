"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""
import os
import torch
from test_api import test_api
from test_api import RobertaEmbeddings
from transformers import RobertaConfig, RobertaTokenizer
from models.modeling_roberta import RobertaModel
import numpy as np
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='SST-2', type=str, help='[SST-2, Yelp, AGNews, TREC, MRPC, SNLI]')
parser.add_argument("--cuda", default=0, type=int)
parser.add_argument("--suffix", default='', type=str)
args = parser.parse_args()

task_name = args.task_name
device = 'cuda:{}'.format(args.cuda)
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

pre_pre_str = tokenizer.decode(list(range(1000, 1050))) + ' . '
# middle_str = ' ? <mask> .'


for seed in [8, 13, 42, 50, 60]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    best = torch.load(f'./results/{task_name}/{seed}/best.pt').to(device).view(50, -1)

    def sentence_fn(test_data):
        """
        This func can be a little confusing.
        Since there are 2 sentences in MRPC and SNLI each sample, we use the same variable `test_data` to represent both.
        test_data is actually a <dummy_token>. It is then replaced by real data in the wrapped API.
        For other 4 tasks, test_data must be used only once, e.g. pre_str + test_data + post_str
        """
        # SST-2: '%s . %s . It was %s .'
        if task_name in ['SST-2', 'Yelp']:
            post_str = ' . It was <mask> .'
            return pre_pre_str + ' . ' + test_data + post_str
        elif task_name == 'AGNews':
            pre_str = ' . <mask> News: '
            return pre_pre_str + pre_str + test_data
        elif task_name == 'TREC':
            pre_str = ' . <mask> question: '
            return pre_pre_str + pre_str + test_data
        elif task_name in ['SNLI', 'MRPC']:
            middle_str = ' ? <mask> , '
            return pre_pre_str + ' . ' + test_data + middle_str + test_data
        
        # return pre_str + test_data + middle_str + test_data


    def embedding_and_attention_mask_fn(embedding, attention_mask):
        # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
        prepad = torch.zeros(size=(1, 1024), device=device)
        pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        return embedding + torch.cat([prepad, best, pospad]), attention_mask
        # return embedding, attention_mask

    predictions = torch.tensor([], device=device)
    for res, _, _ in test_api(
        sentence_fn=sentence_fn,
        embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
        # embedding_and_attention_mask_fn=None,
        test_data_path=f'./test_datasets/{task_name}/encrypted.pth',
        task_name=task_name,
        device=device
    ):
        if task_name in ['SST-2', 'Yelp']:
            c0 = res[:, tokenizer.encode("bad", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("great", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'AGNews':
            c0 = res[:, tokenizer.encode("World", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Sports", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("Business", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("Technology", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'TREC':
            c0 = res[:, tokenizer.encode("description", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("entity", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("abbreviation", add_special_tokens=False)[0]]
            c3 = res[:, tokenizer.encode("human", add_special_tokens=False)[0]]
            c4 = res[:, tokenizer.encode("numeric", add_special_tokens=False)[0]]
            c5 = res[:, tokenizer.encode("location", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2, c3, c4, c5]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'SNLI':
            def change_SNLI_verb(verb_):
                return "No" if verb "Unless"
            c0 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Maybe", add_special_tokens=False)[0]]
            c2 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1, c2]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])
        elif task_name == 'MRPC':
            c0 = res[:, tokenizer.encode("No", add_special_tokens=False)[0]]
            c1 = res[:, tokenizer.encode("Yes", add_special_tokens=False)[0]]
            pred = torch.stack([c0, c1]).argmax(dim=0)
            predictions = torch.cat([predictions, pred])

    save_file_name = task_name + args.suffix
    if not os.path.exists(f'./predictions/{save_file_name}'):
        os.makedirs(f'./predictions/{save_file_name}')
    with open(f'./predictions/{save_file_name}/{seed}.csv', 'w+') as f:
        wt = csv.writer(f)
        wt.writerow(['', 'pred'])
        wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())



