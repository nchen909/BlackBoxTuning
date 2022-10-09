import os
import copy
import time
import math
import pickle
import random
from unittest import loader
from scipy.optimize import minimize
import torch
import argparse
import numpy as np
import cma
from fastNLP import cache_results, Tester, DataSet
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
    BartConfig,
    BartTokenizer,
    T5Config,
    T5Tokenizer,
    GPT2Config,
    GPT2Tokenizer,
)
from models.deep_modeling_roberta import RobertaForMaskedLM
from models.deep_modeling_bart import BartForConditionalGeneration
from models.deep_modeling_t5 import T5ForConditionalGeneration
from models.deep_modeling_gpt2 import GPT2LMHeadModel
from models.deep_modeling_bert import BertForMaskedLM
from models.deep_modeling_cpt import CPTForMaskedLM
from utils import hinge_loss
from sklearn.metrics import f1_score
# from DFO_src.dfo_tr import dfo_tr
from matplotlib.pyplot import savefig
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large',
                    # choices=['roberta-base', 'roberta-large',
                    #          'bert-base-uncased', 'bert-large-uncased',
                    #          'facebook/bart-base', 'facebook/bart-large',
                    #          't5-small', 't5-base', 't5-large', 't5-3b',
                    #          'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                    #          ], 
                    type=str)
parser.add_argument("--model_path", default='roberta-large', type=str, help='The path of hugging face models for offline mode, default=model_name')
parser.add_argument("--task_name", default='TREC', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma1", default=1, type=float)
parser.add_argument("--sigma2", default=0.2, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str, choices=['CMA','openes'])#['CMA','openes','PEPG','DFO_tr']
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)

# for Data Augmentation
parser.add_argument("--data_dir", default='datasets', type=str, help="'dataset' represents origin datasets, for DA, please select 'DA_datasets'. ")

# for CMA
parser.add_argument("--weight_decay", default=0, type=float, help='openes hyperparameter')
# for openes or PEPG
parser.add_argument("--sigma_init", default=2, type=float, help='openes hyperparameter')
parser.add_argument("--learning_rate", default=0.2, type=float, help='openes hyperparameter')
# for two stage DFO, before budget use CMA, and after budget2 use replace_alg
# if use two stage DFO, budget2 be as large as possible, you can choose 6000 or larger for nonCMA seems slow
parser.add_argument("--budget2", default=0, type=int, help='two stage hyperparameter,0 refers not to use')
parser.add_argument("--replace_alg", default='COBYLA', type=str, choices=['Powell','Nelder-Mead','COBYLA','SLSQP','L-BFGS-B'])
# for replace best result0 with best result5 to eval dev, mean solution, presumably better with noise
parser.add_argument("--pop_mean", default=0, type=int, choices=[0,1], help='0:use result.xbest, 1:use result[5], CMA replace result[0] with mean solution to eval dev, presumably better with noise')
parser.add_argument("--calibration", default=0, type=int, choices=[0,1], help='calibration logits for only AGNews,TREC,SST-2, before use please and N/A to trainset and clear cache!')

args = parser.parse_args()

# below are free hyper-params
model_name = args.model_name
if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    from dataloader_t5 import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader
    from metrics_t5 import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    from dataloader_gpt import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader
    from metrics_gpt import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric
else:
    from dataloader import SST2Loader, AGNewsLoader, YelpPLoader, MRPCLoader, SNLILoader, TRECLoader
    from metrics import SST2Metric, AGNewsMetric, YelpPMetric, MRPCMetric, SNLIMetric, TRECMetric

model_path = args.model_path
task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
bound = args.bound
sigma1 = args.sigma1
sigma2 = args.sigma2
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path
save_hiddens = True
data_dir = args.data_dir

budget2 = args.budget2
replace_alg = args.replace_alg
calibration =args.calibration

print("###################################")
#for CMA
if alg == 'CMA':
    weight_decay = args.weight_decay
    print("alg:", alg)
    print("weight_decay:", weight_decay)

# for openes or PEPG
if alg != 'CMA':
    sigma_init = args.sigma_init
    learning_rate = args.learning_rate
    print("alg:", alg)
    print("sigma init, learning rate:", sigma_init, learning_rate)

pop_mean = args.pop_mean
# if pop_mean:
#     budget *= 1 + 1 / (eval_every//popsize - 1)
#     budget = int(budget)

print("pop_mean:", pop_mean)
print("budget:", budget)
print("budget2:", budget2)
print("print_every:", print_every)
print("eval_every:", eval_every)
print("###################################")
# # fixed hyper-params
# if cat_or_add == 'add':
#     init_prompt_path = None
# else:
#     init_prompt_path = './nli_base_prompt.pt'

if task_name in ['SST-2', 'Yelp', 'MRPC']:
    num_labels = 2
elif task_name in ['SNLI']:
    num_labels = 3
elif task_name in ['AGNews']:
    num_labels = 4
elif task_name in ['TREC']:
    num_labels = 5
else:
    raise ValueError

args.bbt_version = 'deepbbt'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class LMForwardAPI:
    def __init__(self, model_name='roberta-large', model_path='roberta-large', n_prompt_tokens=50, task_name='SST-2',
                 loss_type='hinge'):
        self.model_name = model_name
        self.model_path = model_path
        if model_name in ['roberta-base', 'roberta-large']:
            self.config = RobertaConfig.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework=inference_framework,
                onnx_model_path=onnx_model_path,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.config = BertConfig.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
            self.config = BartConfig.from_pretrained(model_path)
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
            self.config = T5Config.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            self.config = GPT2Config.from_pretrained(model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        elif model_name in ['fnlp/cpt-large']:
            self.config = BartConfig.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = CPTForMaskedLM.from_pretrained(
                model_path,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            raise NotImplementedError

        if random_proj == 'normal':
            self.config.output_hidden_states = True

        if inference_framework == 'ort':
            self.model.roberta = None
        self.best_prefix = torch.zeros(self.config.num_hidden_layers, n_prompt_tokens, self.config.hidden_size,
                                       device=device)
        self.best = None
        self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False) for _ in
             range(self.config.num_hidden_layers)])
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['roberta-base', 'roberta-large']:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
                embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['facebook/bart-base', 'facebook/bart-large', 'fnlp/cpt-large']:
                embedding = self.model.model.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                embedding = self.model.transformer.get_input_embeddings().weight.clone().cpu()
            else:  # T5
                embedding = self.model.get_input_embeddings().weight.clone().cpu()
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear[0].parameters():
                torch.nn.init.normal_(p, 0.0, std)
            self.intermediate_stats = [(mu, std)]
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.num_call = 0
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        self.is_better_dev = self.config.num_hidden_layers * [0]
        if task_name == 'SST-2':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'AGNews':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'Yelp':
            self.metric = YelpPMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'MRPC':
            self.metric = MRPCMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'SNLI':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        elif task_name == 'TREC':
            self.metric = TRECMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'TRECMetric'
        else:
            raise NotImplementedError
        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    # def get_p_content_free(params, train_sentences, train_labels, content_free_inputs=('N/A',)):
    #     """Query model with content free input, return its prediction probability for each label"""
    #     label_dict = params['label_dict']

    #     all_p_y = []
    #     for content_free_input in content_free_inputs:
    #         prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)

    #         p_y = [0] * len(label_dict)
    #         for i, answers in label_dict.items():
    #             prob = 0
    #             for a in answers:
    #                 prob += np.exp(complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
    #             p_y[i] = prob
    #         all_p_y.append(p_y)

    #     p_y = np.mean(np.array(all_p_y), axis=0)
    #     p_y = p_y / np.sum(p_y) # normalize
    #     return p_y
    def eval_accuracy(self,all_label_probs, test_labels, mode=None, p_cf=None):
        # evaluate the accuracy with and without contextual calibration
        num_classes = all_label_probs.shape[1]
        if p_cf is None:
            # do not calibrate
            W = np.identity(num_classes)
            b = np.zeros([num_classes, 1])
        else:
            # calibrate
            if mode == "diagonal_W":
                W = np.linalg.inv(np.identity(num_classes) * p_cf)
                b = np.zeros([num_classes, 1])
            elif mode == "identity_W":
                W = np.identity(num_classes)
                b = -1 * np.expand_dims(p_cf, axis=-1)
            else:
                assert False

        correctness_list = []
        assert len(all_label_probs) == len(test_labels)
        for label_probs, true_label in zip(all_label_probs, test_labels):
            label_probs = label_probs / np.sum(label_probs) # normalize to 1

            calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

            ans_label = np.argmax(calibrate_label_probs)
            if ans_label == true_label:
                correctness_list.append(1)
            else:
                correctness_list.append(0)
        return np.mean(correctness_list)

    # # return a tensor each row normalize to 1
    # def normalize(self, x):
    #     return x / x.sum(dim=1, keepdim=True)
    def calc_metric_calibration(self, logits, target, mode="diagonal_W", p_cf=None):
        label_map = self.metric.label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred_old = logits.argmax(dim=-1)
        print("###before calibration pred:", pred_old)
        num_classes = len(label_map)
        # calibrate
        if mode == "diagonal_W":
            W = torch.linalg.inv(torch.eye(num_classes, device=device) * p_cf)
            b = torch.zeros([num_classes, 1], device=device)
        elif mode == "identity_W":
            W = torch.eye(num_classes, device=device)
            b = -1 * torch.unsqueeze(p_cf, axis=-1, device=device)# [num_classes]2[num_classes,1]
        else:
            assert False
        
        
        assert len(logits) == len(converted_target)
        norm_logits = logits / logits.sum(dim=1, keepdim=True) # normalize to 1 , [datasets_len,num_classes]

        calibrate_logits = torch.matmul(norm_logits,W) + b.repeat([1,len(norm_logits)]).t()
        #([num_classes,num_classes]*[num_classes,datasets_len]+[num_classes,datasets_len]).T
        pred = torch.argmax(calibrate_logits,axis=1)
        print("###after calibration pred:", pred)
        if self.metric_key == 'acc':
            perf_old = (pred_old == converted_target).sum() / len(target)
            print("###before calibration perf:", perf_old)
            perf = (pred == converted_target).sum() / len(target)
            print("###after calibration perf:", perf)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map #SNLI: {tokenize:label}:{38861: 0, 40926: 1, 33757: 2}

        converted_target = target.clone()#[datasize]
        for key, val in label_map.items():
            converted_target[target == key] = val #tensor,size=data,[true_label,...]
        interest_index = list(label_map.keys()) #[verbalizer index]
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1) #tensor,[datasize,label_num] ->argmax-> [datasize]

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf

    def get_p_cf(self, prompt_embedding=None, layer_id=None, test_data=None):
        best_prefix = self.best_prefix.clone()
        if prompt_embedding is not None:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear[layer_id](prompt_embedding).reshape(-1, self.config.hidden_size)  # Az
            best_prefix[layer_id] = prompt_embedding

        self.model.set_prompt_embedding(best_prefix)

        for k, v in content_free_data.items():
            content_free_data[k] = v.to(device)
        with torch.no_grad():
            if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                outputs = self.model(
                    input_ids=content_free_data['input_ids'],
                    attention_mask=content_free_data['attention_mask'],
                    decoder_input_ids=content_free_data['decoder_input_ids'],
                    decoder_attention_mask=content_free_data['decoder_attention_mask'],
                )
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                outputs = self.model(
                    input_ids=content_free_data['input_ids'],
                    attention_mask=content_free_data['attention_mask'],
                )
            else:
                outputs = self.model(
                    input_ids=content_free_data['input_ids'],
                    attention_mask=content_free_data['attention_mask'],
                    mask_pos=content_free_data['mask_pos'],
                )
            logits = outputs['logits']
            label_map = self.metric.label_map
            interest_index = list(label_map.keys())
            logits = logits[:,interest_index]
            logits = logits / logits.sum(dim=1, keepdim=True) 
            return logits[0]

    def eval(self, prompt_embedding=None, layer_id=None, test_data=None, p_cf=None):
        self.num_call += 1
        best_prefix = self.best_prefix.clone()
        if prompt_embedding is not None:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear[layer_id](prompt_embedding).reshape(-1, self.config.hidden_size)  # Az
            best_prefix[layer_id] = prompt_embedding

        self.model.set_prompt_embedding(best_prefix)

        for k, v in train_data.items():
            train_data[k] = v.to(device)
        with torch.no_grad():
            if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    decoder_input_ids=train_data['decoder_input_ids'],
                    decoder_attention_mask=train_data['decoder_attention_mask'],
                )
            elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                )
            else:
                outputs = self.model(
                    input_ids=train_data['input_ids'],
                    attention_mask=train_data['attention_mask'],
                    mask_pos=train_data['mask_pos'],
                )
            logits = outputs['logits']
            if random_proj == 'normal' and len(self.intermediate_stats) == 1:
                # if is the first forward pass, record the range of hidden states of each layer
                print('Calculating std for random projections...')
                if self.model_name in ['facebook/bart-base', 'facebook/bart-large',
                                        't5-small', 't5-base', 't5-large', 't5-3b',
                                        'fnlp/cpt-large',
                                        ]:
                    hidden_states = outputs['encoder_hidden_states']
                else:
                    hidden_states = outputs['hidden_states']
                for i, h in enumerate(hidden_states[1:-1]):
                    if save_hiddens:
                        hid_path = './hidstates/{}'.format(self.model_name.split('/')[-1])
                        if not os.path.exists(hid_path):
                            os.makedirs(hid_path, exist_ok=True)
                        with open('{}/hidden_{}.bin'.format(hid_path, i + 1), 'wb') as f:
                            pickle.dump(h, f)
                    print('[Layer {}]'.format(i + 1))
                    hidden = h.clone().reshape(-1).detach().cpu().numpy()
                    mu_hat = np.mean(hidden)
                    std_hat = np.std(hidden)
                    max_h = np.max(hidden)
                    min_h = np.min(hidden)
                    print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                        mu_hat, std_hat, min_h, max_h))
                    # Clipping outliers
                    clip_round = 0
                    while clip_round < 5:
                        clip_round += 1
                        min_bound = mu_hat - 3 * std_hat
                        max_bound = mu_hat + 3 * std_hat
                        hidden = np.clip(hidden, min_bound, max_bound)
                        mu_hat = np.mean(hidden)
                        std_hat = np.std(hidden)
                        max_h = np.max(hidden)
                        min_h = np.min(hidden)
                        print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                            clip_round, mu_hat, std_hat, min_h, max_h))
                    # Calculating std dev for the random projection
                    mu = 0.0
                    std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
                    print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
                    for p in self.linear[i + 1].parameters():
                        torch.nn.init.normal_(p, mu, std)
                    self.intermediate_stats.append((mu, std))
                assert len(self.intermediate_stats) == self.config.num_hidden_layers
                self.model.config.output_hidden_states = None
                print('Random projections initialized.')

            loss, perf = self.calc_metric(logits, train_data['labels']) if type(p_cf)!=type(torch.Tensor([0])) else \
            self.calc_metric_calibration(logits, train_data['labels'], mode="diagonal_W", p_cf=p_cf)

            if perf > self.best_train_perf:
                self.best_train_perf = perf

            if self.num_call % self.print_every == 0 or type(p_cf)==type(torch.Tensor([0])):
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0 or type(p_cf)==type(torch.Tensor([0])):
                print('********* Evaluated on dev set *********')
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                            decoder_input_ids=dev_data['decoder_input_ids'],
                            decoder_attention_mask=dev_data['decoder_attention_mask'],
                        )['logits']
                    elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                        )['logits']
                    else:
                        logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                            mask_pos=dev_data['mask_pos'],
                        )['logits']

                dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels']) if type(p_cf)!=type(torch.Tensor([0])) else \
                self.calc_metric_calibration(logits, dev_data['labels'], mode="diagonal_W", p_cf=p_cf)
                # better_dev_policy = dev_perf >= self.best_dev_perf if pop_mean else dev_perf > self.best_dev_perf
                better_dev_policy = dev_perf > self.best_dev_perf
                if better_dev_policy:
                    self.is_better_dev[layer_id] = 1
                    self.best_dev_perf = dev_perf
                    self.best = best_prefix.clone()#torch.Size([24, 50, 1024])
                    print("###self.num_call:",self.num_call)
                    print("###dev_perf > self.best_dev_perf,dev_perf ,self.best_dev_perf:",dev_perf ,self.best_dev_perf)
                
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
            return loss

    def refresh_is_better_dev(self, layer_id = None):
        self.is_better_dev[layer_id] = 0
        return self.is_better_dev[layer_id]
    
    def calculate_CM(self, best_prompt_embedding=None):
        # self.num_call += 1
        best_prefix = self.best_prefix.clone()
        for i in range(24):
            best_prompt_embedding[i] = torch.tensor(best_prompt_embedding[i]).type(torch.float32)  # z
            best_prompt_embedding[i] = self.linear[i](best_prompt_embedding[i]).reshape(-1, self.config.hidden_size)  # Az
            best_prefix[i] = best_prompt_embedding[i]

        self.model.set_prompt_embedding(best_prefix)

        for k, v in train_data.items():
            train_data[k] = v.to(device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=train_data['input_ids'],
                attention_mask=train_data['attention_mask'],
                mask_pos=train_data['mask_pos'],
            )
            logits = outputs['logits']
            if random_proj == 'normal' and len(self.intermediate_stats) == 1:
                # if is the first forward pass, record the range of hidden states of each layer
                print('Calculating std for random projections...')
                if self.model_name[18:] in ['facebook/bart-base', 'facebook/bart-large',
                                        't5-small', 't5-base', 't5-large', 't5-3b',
                                        'fnlp/cpt-large',
                                        ]:
                    hidden_states = outputs['encoder_hidden_states']
                else:
                    hidden_states = outputs['hidden_states']
                for i, h in enumerate(hidden_states[1:-1]):
                    if save_hiddens:
                        hid_path = './hidstates/{}'.format(self.model_name.split('/')[-1])
                        if not os.path.exists(hid_path):
                            os.makedirs(hid_path, exist_ok=True)
                        with open('{}/hidden_{}.bin'.format(hid_path, i + 1), 'wb') as f:
                            pickle.dump(h, f)
                    print('[Layer {}]'.format(i + 1))
                    hidden = h.clone().reshape(-1).detach().cpu().numpy()
                    mu_hat = np.mean(hidden)
                    std_hat = np.std(hidden)
                    max_h = np.max(hidden)
                    min_h = np.min(hidden)
                    print(' - Before clipping: mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                        mu_hat, std_hat, min_h, max_h))
                    # Clipping outliers
                    clip_round = 0
                    while clip_round < 5:
                        clip_round += 1
                        min_bound = mu_hat - 3 * std_hat
                        max_bound = mu_hat + 3 * std_hat
                        hidden = np.clip(hidden, min_bound, max_bound)
                        mu_hat = np.mean(hidden)
                        std_hat = np.std(hidden)
                        max_h = np.max(hidden)
                        min_h = np.min(hidden)
                        print(' - After clipping (round %d): mu=%.4f, std=%.4f, min=%.4f, max=%.4f' % (
                            clip_round, mu_hat, std_hat, min_h, max_h))
                    # Calculating std dev for the random projection
                    mu = 0.0
                    std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma1)
                    # temp = intrinsic_dim - std_hat * std_hat
                    # mu = mu_hat / temp
                    # std = std_hat / np.sqrt(temp)
                    print(' - Random Projection: mu=%.4f, std=%.4f' % (mu, std))
                    for p in self.linear[i + 1].parameters():
                        torch.nn.init.normal_(p, mu, std)
                    self.intermediate_stats.append((mu, std))
                assert len(self.intermediate_stats) == self.config.num_hidden_layers
                self.model.config.output_hidden_states = None
                print('Random projections initialized.')

        print('********* Evaluated on dev set *********')
        for k, v in dev_data.items():
            dev_data[k] = v.to(device)
        with torch.no_grad():
            logits = self.model(
                input_ids=dev_data['input_ids'],
                attention_mask=dev_data['attention_mask'],
                mask_pos=dev_data['mask_pos'],
            )['logits']

        dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])


if model_name in ['roberta-base', 'roberta-large']:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
elif model_name in ['bert-base-uncased', 'bert-large-uncased', 'fnlp/cpt-large']:
    tokenizer = BertTokenizer.from_pretrained(model_path)
elif model_name in ['facebook/bart-base', 'facebook/bart-large']:
    tokenizer = BartTokenizer.from_pretrained(model_path)
elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
else:
    raise NotImplementedError

# cache_fn = f"caches/data_{model_name.replace('/', '-')}_{data_dir}_{task_name}_{n_prompt_tokens}_{seed}.pt"

DataLoader = {
    'SST-2': SST2Loader,
    'AGNews': AGNewsLoader,
    'Yelp': YelpPLoader,
    'MRPC': MRPCLoader,
    'SNLI': SNLILoader,
    'TREC': TRECLoader,
}


# @cache_results(cache_fn, _refresh=True)
def get_data(task_name, tokenizer):
    splits = ['train', 'dev']
    global data_loader
    data_loader = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens, data_dir=data_dir)
    data_bundle = data_loader.my_load(splits, seed)
    return data_bundle

data_bundle = get_data(task_name=task_name, tokenizer=tokenizer)

train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')
for ds in [train_data, dev_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)


print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])
if calibration or data_loader.is_content_free:
    content_free_data = train_data[-1]
    print('content free data:')
    print(content_free_data)
    train_data= train_data[:-1]
if model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'decoder_input_ids': torch.tensor(train_data['decoder_input_ids'].get(list(range(len(train_data))))),
        'decoder_attention_mask': torch.tensor(train_data['decoder_attention_mask'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'decoder_input_ids': torch.tensor(dev_data['decoder_input_ids'].get(list(range(len(dev_data))))),
        'decoder_attention_mask': torch.tensor(dev_data['decoder_attention_mask'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }
    if calibration:
        content_free_data = {
            'input_ids': torch.tensor(content_free_data['input_ids']),
            'attention_mask': torch.tensor(content_free_data['attention_mask']),
            'decoder_input_ids': torch.tensor(content_free_data['decoder_input_ids']),
            'decoder_attention_mask': torch.tensor(content_free_data['decoder_attention_mask']),
            'labels': torch.tensor(content_free_data['labels']),
        }
elif model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }
    if calibration:
        content_free_data = {
            'input_ids': torch.tensor(content_free_data['input_ids']),
            'attention_mask': torch.tensor(content_free_data['attention_mask']),
            'labels': torch.tensor(content_free_data['labels']),
        }
else:
    train_data = {
        'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
        'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
        'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
        'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
    dev_data = {
        'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
        'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
        'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
        'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
    }
    if calibration:
        content_free_data = {
            'input_ids': torch.unsqueeze(torch.tensor(content_free_data['input_ids']), axis=0),
            'attention_mask': torch.unsqueeze(torch.tensor(content_free_data['attention_mask']), axis=0),
            'mask_pos': torch.unsqueeze(torch.tensor(content_free_data['mask_pos']), axis=0),
            'labels': torch.unsqueeze(torch.tensor(content_free_data['labels']), axis=0),
        }

model_forward_api = LMForwardAPI(
    model_name=model_name,
    model_path=model_path,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    loss_type=loss_type,
)


# cma_opts = {
#     'seed': seed,
#     'popsize': popsize,
#     'maxiter': budget // (popsize * model_forward_api.config.num_hidden_layers),
#     'verbose': -1,
# }
# if bound > 0:
#     cma_opts['bounds'] = [-1 * bound, 1 * bound]

# sigmas = [sigma1]
# for i in range(model_forward_api.config.num_hidden_layers - 1):
#     sigmas.append(sigma2)
# assert len(sigmas) == model_forward_api.config.num_hidden_layers
# es_list = [
#     cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigmas[i], inopts=cma_opts)
#     for i in range(model_forward_api.config.num_hidden_layers)
# ]
# start_time = time.time()




if alg.lower()=='cma':
    cma_or_formalcma='cma'# CMA by import cma or es 
    if cma_or_formalcma=='cma':
        cma_opts = {
            'seed': seed,
            'popsize': popsize,# every popsize times api call ask&tell once
            #分化出popsize=20个种群(x)依次训 所以solutions大小intrinsic_dim*popsize=500*20
            'maxiter': budget // (popsize * model_forward_api.config.num_hidden_layers),
            'verbose': -1,
        }
        if bound > 0:
            cma_opts['bounds'] = [-1 * bound, 1 * bound]
        sigmas = [sigma1]
        for i in range(model_forward_api.config.num_hidden_layers - 1):
            sigmas.append(sigma2)
        assert len(sigmas) == model_forward_api.config.num_hidden_layers
        es_list = [
            cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigmas[i], inopts=cma_opts)
            for i in range(model_forward_api.config.num_hidden_layers)
        ]
        print('Population Size: {}'.format(es_list[0].popsize))

        # opt = cma.CMAOptions()
        start_time = time.time()

        # es.optimize(model_forward_api.eval)
        # #this func executes the same way the loop below, and calls lots time of eval api
        result = model_forward_api.config.num_hidden_layers * [0]
        best_result = model_forward_api.config.num_hidden_layers * [0]
        for round_ in range(budget // (int(popsize) * model_forward_api.config.num_hidden_layers)):
            for i, es in enumerate(es_list):# i refers to layer_num, 0->24 0>24 ...
                print("layer i:",i)
                # update_CMA_per = eval_every - popsize if pop_mean else eval_every
                update_CMA_per = eval_every
                solutions = es.ask()
                #ask delivers new candidate solutions and tell updates the optim instance by passing the respective function values
                fitnesses = [model_forward_api.eval(x, i) for x in solutions]
                #every time we call eval (per 20 popsize), print loss
                #serial模式，20个种群都输入fitnesses
                es.tell(solutions, fitnesses)
                result[i] = es.result#[0]:solution [1]:best f value [2]
                print("i,result[i][1]:",i,result[i][1])
                print("fitness_list[0]:",fitnesses[0])
                # model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i].xbest).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
                #ndarray(500,1) to torch.Size([51200]) to torch.Size([50, 1024])

                if model_forward_api.num_call % eval_every >= update_CMA_per: #eval by result[5] but not update
                    fitness_list_result5 = np.zeros(popsize)
                    for pop_ in range(popsize):
                        if pop_==0:
                            print("eval by using result[5]")
                        fitness_list_result5[pop_] = model_forward_api.eval(result[i][5], i)#just for watch result[5] fitness
                    # model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i][5]).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
                
                if round_==0:
                    print("###i,is_better_dev:",i,model_forward_api.is_better_dev[i])
                    best_result[i] = result[i][5] if pop_mean else result[i][0]
                    model_forward_api.refresh_is_better_dev(i)
                else:
                    if model_forward_api.is_better_dev[i]:
                        print("###i,is_better_dev:",i,model_forward_api.is_better_dev[i])
                        best_result[i] = result[i][5] if pop_mean else result[i][0]
                        model_forward_api.refresh_is_better_dev(i)
                    
                if pop_mean:
                    model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i][5]).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
                else:
                    model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i][0]).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
                if i==len(es_list)-1:
                    es.logger.add()  # write data to disc to be plotted
                    es.disp()
        es_list[-1].logger.plot()
        savefig("./cmaplot.png")
    elif cma_or_formalcma=='formalcma':
        cma_opts = {
            'seed': seed,
            'popsize': popsize,# every popsize times api call ask&tell once
            #分化出popsize=20个种群(x)依次训 所以solutions大小intrinsic_dim*popsize=500*20
            'maxiter': budget // (popsize * model_forward_api.config.num_hidden_layers),
            'verbose': -1,
        }
        if bound > 0:
            cma_opts['bounds'] = [-1 * bound, 1 * bound]
        sigmas = [sigma1]
        for i in range(model_forward_api.config.num_hidden_layers - 1):
            sigmas.append(sigma2)
        assert len(sigmas) == model_forward_api.config.num_hidden_layers
        es_list = [
            cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigmas[i], inopts=cma_opts)
            for i in range(model_forward_api.config.num_hidden_layers)
        ]
        print('Population Size: {}'.format(es_list[0].popsize))
        start_time = time.time()

        for _ in range(budget // (int(popsize) * model_forward_api.config.num_hidden_layers)):
            for i, es in enumerate(es_list):
                solutions = es.ask()
                fitnesses = [model_forward_api.eval(x, i) for x in solutions]
                es.tell(solutions, fitnesses)
                model_forward_api.best_prefix[i] = model_forward_api.linear[i](
                    torch.tensor(es.result.xbest).type(torch.float32)).reshape(-1,
                                                                            model_forward_api.config.hidden_size)  # set best cv
            if _  % 2 == 0:
                epoch_num = _ * (int(popsize) * model_forward_api.config.num_hidden_layers)
                if not os.path.exists(f'./results/{epoch_num}_results/{task_name}/{seed}'):
                    os.makedirs(f'./results/{epoch_num}_results/{task_name}/{seed}')
                torch.save(model_forward_api.best_prefix, f=f'./results/{epoch_num}_results/{task_name}/{seed}/best.pt')


        end_time = time.time()
        print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
        if not os.path.exists(f'./results/last_epoch_results/{task_name}/{seed}'):
            os.makedirs(f'./results/last_epoch_results/{task_name}/{seed}')

        if not os.path.exists(f'./results/dev_best_results/{task_name}/{seed}'):
            os.makedirs(f'./results/dev_best_results/{task_name}/{seed}')

        torch.save(model_forward_api.best, f=f'./results/dev_best_results/{task_name}/{seed}/best.pt')
        torch.save(model_forward_api.best_prefix, f=f'./results/last_epoch_results/{task_name}/{seed}/best.pt')
elif alg.lower() in ['openes']:
    import es
    #solver = es.SimpleGA(...)
    #solver = es.PEPG(...)
    #solver = es.OpenES(...)
    #solver = es.CMAES(...)
    # solver =es.CMAES(intrinsic_dim,      # number of model parameters
    #            sigma_init=1,       # initial standard deviation
    #            popsize=popsize           # population size
    #         #    weight_decay=0.01    # weight decay coefficient)
    #         )

    sigmas = [sigma1]
    for i in range(model_forward_api.config.num_hidden_layers - 1):
        sigmas.append(sigma2)
    assert len(sigmas) == model_forward_api.config.num_hidden_layers
    es_list = [
        es.OpenES(intrinsic_dim,                  # number of model parameters
        seed=seed,
        sigma_init=sigmas[i],            # initial standard deviation
        sigma_decay=1,         # don't anneal standard deviation
        learning_rate=learning_rate,         # learning rate for standard deviation
        learning_rate_decay = 1, # annealing the learning rate
        popsize=popsize,       # population size
        antithetic=False,          # whether to use antithetic sampling
        weight_decay=0,         # weight decay coefficient
        rank_fitness=False,        # use rank rather than fitness numbers
        forget_best=False)         # forget historical best, for output
        for i in range(model_forward_api.config.num_hidden_layers)
    ]

    start_time = time.time()

    result = model_forward_api.config.num_hidden_layers * [0]
    best_result = model_forward_api.config.num_hidden_layers * [0]
    for round_ in range(budget // (int(popsize) * model_forward_api.config.num_hidden_layers)):
        for i, es in enumerate(es_list):# i refers to layer_num, 0->24 0>24 ...
            print("layer i:",i)
            # update_CMA_per = eval_every - popsize if pop_mean else eval_every
            update_CMA_per = eval_every
            solutions = es.ask()
            #ask delivers new candidate solutions and tell updates the optim instance by passing the respective function values
            fitnesses = [-model_forward_api.eval(x, i) for x in solutions]
            #every time we call eval (per 20 popsize), print loss
            #serial模式，20个种群都输入fitnesses
            es.tell(fitnesses)
            result[i] = es.result()#[0]:solution [1]:best f value [2]
            print("i,result[i][1]:",i,result[i][1])
            print("fitness_list[0]:",fitnesses[0])
            # model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i].xbest).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
            #ndarray(500,1) to torch.Size([51200]) to torch.Size([50, 1024])

            if round_==0:
                print("###i,is_better_dev:",i,model_forward_api.is_better_dev[i])
                best_result[i] = result[i][5] if pop_mean else result[i][0]
                model_forward_api.refresh_is_better_dev(i)
            else:
                if model_forward_api.is_better_dev[i]:
                    print("###i,is_better_dev:",i,model_forward_api.is_better_dev[i])
                    best_result[i] = result[i][5] if pop_mean else result[i][0]
                    model_forward_api.refresh_is_better_dev(i)
            model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(result[i][0]).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv

if budget2:
    cycle_layer = "large"
    if cycle_layer == "small":
        print("finish CMA. start 2nd stage DFO")
        # use 2 stage because intrinsic_dim=500 is to large for classical none evolution algorithm too converge and it needs to many api to call (per iter)
        for round_ in range(budget2 // ( 20 * model_forward_api.config.num_hidden_layers)):
            for i in range(model_forward_api.config.num_hidden_layers):# i refers to layer_num #0->..->0,...,24-->..->24
                # dev advance quickly when layers num is 0,1
                print("layer i:",i)
                max_iter=budget2 // model_forward_api.config.num_hidden_layers
                def short_api(x_):#[500,1] to [20]
                    fitness = model_forward_api.eval(x_,i)
                    return fitness

                options={}
                for methods in ['Powell','Nelder-Mead']:#, "maxfev": 100 powell do not need seed
                    options[methods]={"disp": True, "maxiter":1}
                    #use maxfev not callback, maxiter callback (only powell has fev)
                    #maxfev just call eval maxfev times, maxiter not match budget2 cause Powell needs call eval many times one fev
                    #if there is not enough one iter, Powell will not update x
                    #you can not know how much fev or call one iter so you cant update one layer by one layer per call
                    #near 12*intrinsic_dim fev per iter, than callback
                    
                    #the most naive non gradient algotithm Nelder-Mead cracks in this problem

                    #but you can use maxiter=1 in COBYLA
                for methods in ['BFGS','SLSQP','L-BFGS-B']:#, "maxiter": 80/200
                    options[methods]={"disp": True, "maxiter":max_iter}
                for methods in ['COBYLA']:
                    options[methods]={"disp": False,"maxiter":20}
                # options['Powell'].update({"ftol": 1e-26}) 
                # options['Nelder-Mead'].update({"ftol": 1e-26})
                # options['COBYLA'].update({"tol": 1e-25})
                # options['BFGS'].update({"gtol": 1e-5})
                # options['SLSQP'].update({"ftol": 1e-26})

                start_time = time.time()
                def callback_f(x_):
                    print("i,fitness:",i,short_api(x_))
                    model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(x_).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
                res = minimize(short_api,  best_result[i],  method=replace_alg, callback=callback_f, options=options[replace_alg])
                best_result[i]= res.x
                print("layer,loss(fitnesses):",i,res.fun)
    elif cycle_layer == "large":
        print("finish CMA. start 2nd stage DFO")

        for i in range(model_forward_api.config.num_hidden_layers):# i refers to layer_num #0->..->0,...,24-->..->24
            print("layer i:",i)
            max_iter=budget2 // model_forward_api.config.num_hidden_layers
            def short_api(x_):#[500,1] to [20]
                fitness = model_forward_api.eval(x_,i)
                return fitness

            options={}
            for methods in ['Powell','Nelder-Mead']:#, "maxfev": 100 powell do not need seed
                options[methods]={"disp": True, "maxfev":max_iter}
            for methods in ['BFGS','SLSQP','L-BFGS-B']:#, "maxiter": 80/200
                options[methods]={"disp": True, "maxiter":max_iter}
            for methods in ['COBYLA']:
                options[methods]={"disp": False,"maxiter":max_iter}
            # options['Powell'].update({"ftol": 1e-26}) 
            # options['Nelder-Mead'].update({"ftol": 1e-26})
            # options['COBYLA'].update({"tol": 1e-25})
            # options['BFGS'].update({"gtol": 1e-5})
            # options['SLSQP'].update({"ftol": 1e-26})

            start_time = time.time()
            def callback_f(x_):
                print('i,fitness(x_):',i,short_api(x_))
                model_forward_api.best_prefix[i] = model_forward_api.linear[i](torch.tensor(x_).type(torch.float32)).reshape(-1,model_forward_api.config.hidden_size)  # set best cv
            
            res = minimize(short_api,  best_result[i],  method=replace_alg,callback=callback_f, options=options[replace_alg])

            print("layer,loss(fitnesses):",i,res.fun)
if budget!=0:
    with open('best_result.pickle', 'wb') as file:
        # A new file will be created
        print("save best_result to best_result.pickle")
        pickle.dump(best_result, file)
    with open('best_prefix.pickle', 'wb') as file:
        # A new file will be created
        print("save best_prefix to best_prefix.pickle")
        pickle.dump(model_forward_api.best_prefix, file)
if calibration:
    print("### start calibration ###")
    with open('best_result.pickle', 'rb') as file:
        best_result = pickle.load(file)
    with open('best_prefix.pickle', 'rb') as file:
        model_forward_api.best_prefix = pickle.load(file)
    p_cf = model_forward_api.config.num_hidden_layers * [0]
    for i in [model_forward_api.config.num_hidden_layers-1]:
        print("layer i :",i)
        p_cf = model_forward_api.get_p_cf(best_result[i], i)
        fitnesses =model_forward_api.eval(best_result[i], i, None, p_cf)
    with open('p_cf.pickle', 'wb') as file:
        # A new file will be created
        print("save p_cf to p_cf.pickle")
        pickle.dump(p_cf, file)
    torch.save(model_forward_api.best, f=f'./results/{task_name}/{seed}/best.pt')
    print("### end calibration ###")
    # elif cma_or_escma=='escma':
    #     import es
    #     solver =es.CMAES(intrinsic_dim,      # number of model parameters
    #                seed=seed,
    #                sigma_init=sigma1,       # initial standard deviation
    #                popsize=popsize,           # population size
    #                weight_decay=weight_decay    # weight decay coefficient)
    #             )
    #     start_time = time.time()
    #     while model_forward_api.num_call < budget:
    #         update_CMA_per = eval_every - popsize if pop_mean else eval_every
    #         if model_forward_api.num_call % eval_every < update_CMA_per:#80:
    #             solutions = solver.ask()
    #             fitness_list = np.zeros(solver.popsize)
    #             for i in range(solver.popsize):
    #                 fitness_list[i] = -model_forward_api.eval(solutions[i])
    #             solver.tell(fitness_list)
    #             solver.logger_add()
    #             solver.disp()
    #             result = solver.result()
    #             if model_forward_api.is_better_dev:
    #                 print("###is_better_dev")
    #                 result0 = result[0]
    #                 result5 = result[5]
    #                 print("result0.shape:",result[0].shape)
    #                 print("result5.shape:",result[5].shape)
    #                 # Open a file and use dump()
    #                 with open('result0.pickle', 'wb') as file:
    #                     # A new file will be created
    #                     pickle.dump(result0, file)
    #                 with open('result5.pickle', 'wb') as file:
    #                     # A new file will be created
    #                     pickle.dump(result5, file)
    #                 model_forward_api.refresh_is_better_dev()
    #         else: #eval by result[5] but not update
    #             fitness_list_result5 = np.zeros(solver.popsize)
    #             for i in range(solver.popsize):
    #                 if i==0:
    #                     print("eval by using result[5]")
    #                 fitness_list_result5[i] = -model_forward_api.eval(result[5])
    #             # mean solution, presumably better with noise
    #     if budget2:
    #         pass
    #                 # solutions = solver.ask()
    #                 # fitness_list = np.zeros(solver.popsize)
    #                 # for i in range(solver.popsize):
    #                 #     if i==0:
    #                 #         print("###eval by using result[5]")
    #                 #     fitness_list[i] = -model_forward_api.eval(result5)
    #                 #     # mean solution, presumably better with noise
    #                 # solver.tell(fitness_list)
    #                 # solver.logger_add()
    #                 # solver.disp()
    #                 # result = solver.result()

    #         # else: #eval by result[5] but not update
    #         #     fitness_list_result5 = np.zeros(solver.popsize)
    #         #     for i in range(solver.popsize):
    #         #         if i==0:
    #         #             print("eval by using result[5]")
    #         #         fitness_list_result5[i] = -model_forward_api.eval(result[5])
    #         #     # mean solution, presumably better with noise


    #             # result5= result[5]
    #             # solutions = solver.ask()
    #             # fitness_list = np.zeros(solver.popsize)
    #             # for i in range(solver.popsize):
    #             #     if i==0:
    #             #         print("eval by using result[5]")
    #             #     fitness_list[i] = -model_forward_api.eval(result5)
    #             #     # mean solution, presumably better with noise
    #             # solver.tell(fitness_list)
    #             # solver.logger_add()
    #             # solver.disp()
    #             # result = solver.result()
    #         # if result[1] >10:
    #         #     break
    #     # fitness_list_result5 = np.zeros(solver.popsize)
    #     # for j in range(5):
    #     #     for i in range(solver.popsize):
    #     #         if i==0:
    #     #             print("eval by using result[5]")
    #     #         fitness_list_result5[i] = -model_forward_api.eval(result[5])
    #     #         # mean solution, presumably better with noise

    #     print("result[1]:",result[1])
    #     print("fitness_list[0]:",fitness_list[0])
    #     # print("fitness_list_result5[0]:",fitness_list_result5[0])
    #     print("-model_forward_api.eval(result[5]):",-model_forward_api.eval(result[5]))
    #     solver.plot()
    #     savefig("./cmaplot.png")



end_time = time.time()
print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
if not os.path.exists(f'./results/{task_name}/{seed}'):
    os.makedirs(f'./results/{task_name}/{seed}')

torch.save(model_forward_api.best, f=f'./results/{task_name}/{seed}/best.pt')
