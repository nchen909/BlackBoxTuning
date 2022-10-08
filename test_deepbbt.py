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
parser.add_argument("--seed", default=8, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--calibration", default=0, type=int, choices=[0,1], help='calibration logits')
# parser.add_argument("--suffix", default='', type=str)
args = parser.parse_args()

task_name = args.task_name
seed = args.seed
print_every = args.print_every
eval_every = args.eval_every
calibration = args.calibration
model_name = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def sentence_fn_factory(task_name):
    prompt_initialization = tokenizer.decode(list(range(1000, 1050)))
    if task_name in ['MRPC', 'SNLI']:
        def sentence_fn(test_data):
            return prompt_initialization + ' . ' + test_data + ' ? <mask> , ' + test_data
            # return prompt_initialization + ' . ' + test_data + ' , <mask> , no , ' + test_data

    elif task_name == 'SST-2':
        def sentence_fn(test_data):
            return prompt_initialization + ' . ' + test_data + ' . It was <mask> .'

    elif task_name == 'AGNews':
        def sentence_fn(test_data):
            return prompt_initialization + ' . <mask> News: ' + test_data

    elif task_name == 'TREC':
        def sentence_fn(test_data):
            return prompt_initialization + ' . <mask> question:' + test_data

    elif task_name == 'Yelp':
        def sentence_fn(test_data):
            return prompt_initialization + ' . ' + test_data + ' .It was <mask> .'

    else:
        raise ValueError

    return sentence_fn

def change_SNLI_verb(verb_,change_or_not):
    if change_or_not==0:
        return verb_
    if verb_ == 'No':
        return 'Except'#'Unless'
    elif verb_ == 'Maybe':
        return 'Watch'#'Fortunately'
    elif verb_ == 'Yes':
        return 'Alright'#'Regardless'
verbalizer_dict = {
    'SNLI': [change_SNLI_verb('Yes',0), change_SNLI_verb('Maybe',0), change_SNLI_verb('No',0)],
    'MRPC': ["No", "Yes"],
    'SST-2': ["bad", "great"],
    'AGNews': ["World", "Sports", "Business", "Tech"],
    'TREC': ["Description", "Entity", "Abbreviation", "Human", "Numeric", "Location"],
    'Yelp': ["bad", "great"]
}


device = 'cuda:0'


for task_name in [task_name]:  # 'SNLI', 'SST-2', 'MRPC', 'AGNews', 'TREC', 
    for seed in [seed]:  # 8, 13, 42, 50, 
        torch.manual_seed(seed)
        np.random.seed(seed)
        # CM = torch.load(f'./bbtv2_results/{task_name}/{seed}/CM.pt').to(device)
        best_prompt = torch.load(f'./results/{task_name}/{seed}/best.pt').to(device).view(24 , 50, -1)

        sentence_fn = sentence_fn_factory(task_name)
        # def embedding_and_attention_mask_fn(embedding, attention_mask):
        #     # res = torch.cat([init_prompt[:-5, :], input_embed, init_prompt[-5:, :]], dim=0)
        #     prepad = torch.zeros(size=(1, 1024), device=device)
        #     pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
        #     return embedding + torch.cat([prepad, best, pospad]), attention_mask

        def hidden_states_and_attention_mask_fn(i, embedding, attention_mask):
            prepad = torch.zeros(size=(1, 1024), device=device)
            pospad = torch.zeros(size=(embedding.size(1) - 51, 1024), device=device)
            return embedding + torch.cat([prepad, best_prompt[i], pospad]), attention_mask

        predictions = torch.tensor([], device=device)
        for res, _, _ in test_api(
            sentence_fn=sentence_fn,
            # embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
            hidden_states_and_attention_mask_fn=hidden_states_and_attention_mask_fn,
            test_data_path=f'./test_datasets/{task_name}/encrypted.pth',
            task_name=task_name,
            device=device,
            # tokenizer_path=model_name, 
            # model_path=model_name,
        ):

            verbalizers = verbalizer_dict[task_name]
            intrested_logits = [res[:, tokenizer.encode(verbalizer, add_special_tokens=False)[0]] for verbalizer in verbalizers]
            intrested_logits = torch.stack(intrested_logits) #[label_num,data_num]
            pred_before = intrested_logits.argmax(dim=0)
            
            if calibration:
                print("pred_no_calibration:", pred_before)
                intrested_logits = intrested_logits / intrested_logits.sum(dim=0, keepdim=True)
                import pickle
                with open('p_cf.pickle', 'rb') as file:
                    p_cf = pickle.load(file)
                W = torch.linalg.inv(torch.eye(intrested_logits.shape[0], device=device) * p_cf[0].cuda())
                b = torch.zeros([intrested_logits.shape[0], intrested_logits.shape[1]], device=device)
                intrested_logits = torch.matmul(W,intrested_logits) + b
            pred = intrested_logits.argmax(dim=0)
            print("pred:", pred)
            # intrested_logits = torch.softmax(torch.stack(intrested_logits).T, dim=1)
            # intrested_logits = torch.mul(intrested_logits, CM)
            # if args.use_CM:
            # for i in range(len(intrested_logits)):
            #     intrested_logits[i] *= CM[0][i]
            # pred = intrested_logits.argmax(dim=1)
            predictions = torch.cat([predictions, pred])

        if not os.path.exists(f'./predictions/{task_name}'):
            os.makedirs(f'./predictions/{task_name}')
        with open(f'./predictions/{task_name}/{seed}.csv', 'w+') as f:
            wt = csv.writer(f)
            wt.writerow(['', 'pred'])
            wt.writerows(torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())