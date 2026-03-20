import numpy as np
import random
import torch
import argparse


RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'llmrec'


def set_template(args):
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
    ##
        dataset_code = {'h': 'home', '1':'home_category','t': 'toys', '2':'toys_category'}
    ##
        args.dataset_code = dataset_code[input('**t 누르면 toys**, **h 누르면 home**, **1번 누르면 home_category** **2번 누르면 toys_category**\n **train_ranker는 h, t만 누릅시다!**: ')]

    args.item_model_path = f"./experiments/lru/{args.dataset_code}/models/best_acc_model.pth"
    args.category_model_path = f"./experiments/lru/{args.dataset_code}_category/models/best_acc_model.pth"
    args.item_retriever_pth_path = f'experiments/lru/{args.dataset_code}/models/best_acc_model.pth'
    args.category_retriever_pth_path = f'experiments/lru/{args.dataset_code}_category/models/best_acc_model.pth'
    args.item_projector_pth_path = f'experiments/projector/{args.dataset_code}/projector.pth'
    args.category_projector_pth_path = f'experiments/projector/{args.dataset_code}_category/projector.pth'
    args.llm_item_retrieved_path = f'/home/work/sg/LlamaRec-doit/experiments/lru/{args.dataset_code}/retrieved.pkl'
    args.llm_category_retrieved_path = f'/home/work/sg/LlamaRec-doit/experiments/lru/{args.dataset_code}_category/retrieved.pkl'

    args.bert_max_len = 50

    if 'llm' in args.model_code: 
        batch = 16 if args.dataset_code == 'ml-100k' else 12
        args.lora_micro_batch_size = batch
    else: 
        batch = 16 if args.dataset_code == 'ml-100k' else 64

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10,]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None
    args.padding_item_id = 0


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)


## 추가
parser.add_argument('--topk', type=int, default=10, help='Top-k 후보 추출(리랭킹을 위한), get_topk_candidates() -> dataloader/llm.py')
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--llm_hidden_dim', type=int, default=4096)
parser.add_argument("--item_model_path", type=str,default="./experiments/lru/home/models/best_acc_model.pth")
parser.add_argument("--category_model_path", type=str,default="./experiments/lru/home_category/models/best_acc_model.pth")
parser.add_argument('--item_retriever_pth_path', type=str, default='experiments/lru/home/models/best_acc_model.pth')
parser.add_argument('--category_retriever_pth_path', type=str, default='experiments/lru/home_category/models/best_acc_model.pth')
parser.add_argument('--item_projector_pth_path', type=str, default='experiments/projector/home/projector.pth')
parser.add_argument('--category_projector_pth_path', type=str, default='experiments/projector/home_category/projector.pth')


# parser.add_argument('--lru_mode', type=str, default='item', choices=['item', 'category'])
# parser.add_argument('--candidate_num', type=int, default=20, help='top-k 후보 개수')
################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=1) ## 우선 돌아가는거 보기 위해.. 기존=500에폭
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='epoch', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=500)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=20)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10]) ## , 50
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 5, 10])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--rerank_best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=False)

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.2)
parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# LLM Model
################
parser.add_argument('--llm_base_model', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--llm_base_tokenizer', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--llm_max_title_len', type=int, default=32)
parser.add_argument('--llm_max_text_len', type=int, default=1536)
parser.add_argument('--llm_max_history', type=int, default=20)
parser.add_argument('--llm_train_on_inputs', type=bool, default=False)
parser.add_argument('--llm_negative_sample_size', type=int, default=9)  # 19 negative & 1 positive
parser.add_argument('--llm_system_template', type=str,  # instruction
    default="Given user history in chronological order, recommend an item from the candidate pool with its index letter.")
parser.add_argument('--llm_input_template', type=str, \
    default='User history: {}; \n Candidate pool: {}')
parser.add_argument('--llm_load_in_4bit', type=bool, default=True)

##
parser.add_argument('--llm_item_retrieved_path', type=str, default='/home/work/sg/LlamaRec-doit/experiments/lru/home/retrieved.pkl')
parser.add_argument('--llm_category_retrieved_path', type=str, default='/home/work/sg/LlamaRec-doit/experiments/lru/home_category/retrieved.pkl')
##
parser.add_argument('--llm_cache_dir', type=str, default='/home/work/sg/.hf_cache')
##


################
# Lora
################
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'v_proj'])
parser.add_argument('--lora_num_epochs', type=int, default=1)
parser.add_argument('--lora_val_iterations', type=int, default=100)
parser.add_argument('--lora_early_stopping_patience', type=int, default=20)
parser.add_argument('--lora_lr', type=float, default=1e-4)
parser.add_argument('--lora_micro_batch_size', type=int, default=16)

################
# projector
################
parser.add_argument('--lr_decay_min_lr', type=float, default=1e-6)
parser.add_argument('--lr_warmup_start_lr', type=float, default=1e-7)
parser.add_argument('--max_steps', type=int, default=100000)


args = parser.parse_args()

