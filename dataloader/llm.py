from .base import AbstractDataloader
from .utils import Prompter
import argparse
from tqdm import tqdm
import torch
import random
import numpy as np
import torch.utils.data as data_utils
from model.lru import LRURec 
import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks
from model.mlp_projector import MlpProjector
from datasets.toys_category import ToysCategoryDataset
from datasets.home_category import HomeCategoryDataset
from datasets.home import HomeDataset
from .utils import load_split_retrieved 
from functools import partial

from collections import defaultdict
def worker_init_fn(worker_id):
    random.seed(int(np.random.get_state()[1][0]) + worker_id)                                                     
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class DummyDataset(data_utils.Dataset):
    """Dummy dataset to handle empty validation/test cases"""
    def __init__(self):
        pass
    
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # Return minimal dummy data that won't break the training loop
        return {
            'input_ids': [1, 2, 3],  # dummy token ids
            'attention_mask': [1, 1, 1],
            'labels': 0,
            'candidates': [1, 2, 3, 4, 5],
            'category_candidates': [1, 2, 3, 4, 5],
            'item_seq': [1, 2],
            'category_seq': [1, 2]
        }


# the following prompting is based on alpaca
def generate_and_tokenize_eval(args, data_point, tokenizer, prompter):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])
    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)
    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
    return tokenized_full_prompt


def generate_and_tokenize_train(args, data_point, tokenizer, prompter):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.llm_max_text_len,
                           padding=False,
                           return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = prompter.generate_prompt(data_point["system"],
                                           data_point["input"],
                                           data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
    if not args.llm_train_on_inputs:
        tokenized_full_prompt["labels"][:-2] = [-100] * len(tokenized_full_prompt["labels"][:-2])
    
    return tokenized_full_prompt


def seq_to_token_ids(
    args, seq, candidates, label, text_dict, tokenizer, prompter, eval=False,
    category_seq=None, category_candidates=None, category_meta_dict=None
    ):
    print(" [DEBUG] category_candidates =", category_candidates)
    print(" [DEBUG] category_candidates types =", [type(x) for x in category_candidates])

    invalid_keys = [cat for cat in category_candidates if cat not in category_meta_dict]
    print(" [DEBUG] Not in category_meta_dict:", invalid_keys)
    print(f"[DEBUG] category_meta_dict keys (sample): {list(category_meta_dict.keys())[:10]}")
    print(f"[DEBUG] max key: {max(category_meta_dict.keys())}, min key: {min(category_meta_dict.keys())}")
    def truncate_title(title):
        if isinstance(title, dict):
            title = title.get('title', '[NO TITLE]')
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        return tokenizer.convert_tokens_to_string(title_)

    # === 아이템 시퀀스 + 카테고리 시퀀스 표현 ===
    seq_t = ' \n '.join([
        f"({i+1}) " + truncate_title(text_dict[item]) + " [" + category_meta_dict[cat] + "]"
        for i, (item, cat) in enumerate(zip(seq, category_seq))
    ])

    # === 아이템 후보 + 카테고리 후보 표현 ===
    can_t = ' \n '.join([
        f"({chr(ord('A')+i)}) " + truncate_title(text_dict[item]) + " [" + category_meta_dict[cat] + "]"
        for i, (item, cat) in enumerate(zip(candidates, category_candidates))
    ])
    output = chr(ord('A') + candidates.index(label))  # ranking only
    
    data_point = {}
    data_point['system'] = args.llm_system_template if args.llm_system_template is not None else DEFAULT_SYSTEM_PROMPT
    data_point['input'] = args.llm_input_template.format(seq_t, can_t)
    data_point['output'] = output
    
    if eval:
        result= generate_and_tokenize_eval(args, data_point, tokenizer, prompter)
    else:
        result= generate_and_tokenize_train(args, data_point, tokenizer, prompter)
    # projector용 정보 추가
    result.update({
        'candidates': candidates,
        'category_candidates': category_candidates,
        'item_seq': seq,
        'category_seq': category_seq
    })

    return result


class LLMDataloader():
    def __init__(self, args):
        self.args = args
        self.rng = np.random
        
        self.llm_item_dataset_path = args.item_dataset
        self.llm_category_dataset_path = args.category_dataset
        
        with open(self.llm_item_dataset_path, 'rb') as f:
            item_seq_dataset = pickle.load(f)
        with open(self.llm_category_dataset_path, 'rb') as f:
            category_seq_dataset = pickle.load(f)     

        self.item_train = item_seq_dataset['train']
        self.item_val = item_seq_dataset['val']
        self.item_test = item_seq_dataset['test']
        self.item_umap = item_seq_dataset['umap']
        self.item_smap = item_seq_dataset['smap']
        self.item_text_dict = item_seq_dataset['meta']
        self.item_cmap = item_seq_dataset['cmap']

        self.category_train = category_seq_dataset['train']
        self.category_val = category_seq_dataset['val']
        self.category_test = category_seq_dataset['test']
        self.category_umap = category_seq_dataset['umap']
        self.category_smap = category_seq_dataset['smap']
        self.category_text_dict = category_seq_dataset['meta']
        self.category_cmap = category_seq_dataset['cmap']

        args.num_items = len(self.item_smap)
        args.num_categories = len(self.category_cmap)

        asin_to_item = {
            asin: self.item_smap[asin]
            for asin in self.item_smap
            if asin in self.item_text_dict
        }

        asin_to_category = {
            asin: self.category_cmap[self.item_text_dict[asin]['category_label']]
            for asin in asin_to_item
        }

        item_to_category = {
            item_idx: asin_to_category[asin]
            for asin, item_idx in asin_to_item.items()
        }

        category_to_items = {
            cat_idx: [self.item_smap[asin] for asin in entry['sids'] if asin in self.item_smap]
            for cat_idx, entry in self.category_text_dict.items()
        }

        self.item_to_category = item_to_category
        self.category_to_items = dict(category_to_items)
        args.item_to_category = item_to_category
        args.category_to_items = dict(category_to_items)


        
        # self.user_count = len(self.item_umap)
        # self.item_count = len(self.item_smap)
        # self.category_count = len(self.category_cmap)
        
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_base_tokenizer, cache_dir=args.llm_cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.clean_up_tokenization_spaces = True
        self.prompter = Prompter()
        
        self.llm_item_dataset_path = args.llm_item_retrieved_path
        self.llm_category_dataset_path = args.llm_category_retrieved_path
        print('Loading item retrieved file from {}'.format(self.llm_item_dataset_path))
        retrieved_item = pickle.load(open(self.llm_item_dataset_path, 'rb'))
        print('Loading category retrieved file from {}'.format(self.llm_category_dataset_path))        
        retrieved_cat = pickle.load(open(self.llm_category_dataset_path, 'rb'))
        
        # ======================= ITEM RETRIEVAL SETUP =======================
        print('# ======================= ITEM RETRIEVAL SETUP =======================')
        self.item_val_probs = retrieved_item['val_probs']
        self.item_val_labels = retrieved_item['val_labels']
        self.item_test_probs = retrieved_item['test_probs']
        self.item_test_labels = retrieved_item['test_labels']
        self.item_val_metrics = retrieved_item['val_metrics']
        self.item_test_metrics = retrieved_item['test_metrics']

        # ======================= GENERATE VAL CANDIDATES =======================
        print('# ======================= GENERATE VAL CANDIDATES =======================')
        self.item_val_users = []
        self.item_val_candidates = []
        self.item_based_category_val_candidates = []

        for u, (p, l) in tqdm(enumerate(zip(self.item_val_probs, self.item_val_labels), start=1), desc="VAL Candidates"):
            cat = self.item_to_category.get(l, None)
            if cat is None:
                continue
            item_cands, cat_cands = self._generate_10_candidates(
                user_probs=torch.tensor(p),
                answer_item=l,
                answer_category=cat,
                item_to_category=self.item_to_category,
                category_to_items=self.category_to_items,
                category_meta_dict=self.category_text_dict
            )
            
            if l in item_cands:
                print(f"[OK VAL] user={u} label={l} in item_cands={item_cands[:3]}... total={len(item_cands)}")
                self.item_val_users.append(u)
                self.item_val_candidates.append(item_cands)
                self.item_based_category_val_candidates.append(cat_cands)
            else:
                print(f"[SKIP VAL] user={u} label={l} NOT in item_cands={item_cands[:3]}... total={len(item_cands)}")
        print(f"[DEBUG] Final item_val_users: {len(self.item_val_users)}")

        # ======================= GENERATE TEST CANDIDATES =======================
        print('# ======================= GENERATE TEST CANDIDATES =======================')
        self.item_test_users = []
        self.item_test_candidates = []
        self.item_based_category_test_candidates = []
        self.item_based_category_test_users = []
        self.item_non_test_users = []

        for u, (p, l) in tqdm(enumerate(zip(self.item_test_probs, self.item_test_labels), start=1), desc="TEST Candidates"):
            cat = self.item_to_category.get(l, None)


            if cat is None:
                continue
            
            print(f"[DEBUG] Sample val_labels: {self.item_val_labels[:5]}")
            print(f"[DEBUG] Max item index in val_labels: {max(self.item_val_labels)}")
            print(f"[DEBUG] user_probs length: {len(self.item_val_probs[0])}")
            item_cands, cat_cands = self._generate_10_candidates(
                user_probs=torch.tensor(p),
                answer_item=l,
                answer_category=cat,
                item_to_category=self.item_to_category,
                category_to_items=self.category_to_items,
                category_meta_dict=self.category_text_dict
            )
            print(f"[DEBUG] answer_item: {l}")
            print(f"[DEBUG] generated item_cands: {item_cands}")
            if l not in item_cands:
                print(f"[WARNING] answer_item {l} NOT in item_cands")            
            if l in item_cands:
                self.item_test_users.append(u)
                self.item_test_candidates.append(item_cands)
                self.item_based_category_test_users.append(u)
                self.item_based_category_test_candidates.append(cat_cands)
            else:
                self.item_non_test_users.append(u)

        # ======================= Build item_test_retrieval dict =======================
        self.item_test_retrieval = {
            'original_size': len(self.item_test_probs),
            'retrieval_size': len(self.item_test_candidates),
            'original_metrics': self.item_test_metrics,
            'retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.item_test_probs)[torch.tensor(self.item_test_users, dtype=torch.long) - 1],
                torch.tensor(self.item_test_labels)[torch.tensor(self.item_test_users, dtype=torch.long) - 1],
                self.args.metric_ks,
            ),
            'non_retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.item_test_probs)[torch.tensor(self.item_non_test_users, dtype=torch.long) - 1],
                torch.tensor(self.item_test_labels)[torch.tensor(self.item_non_test_users, dtype=torch.long) - 1],
                self.args.metric_ks,
            ),
        }

        # ======================= CATEGORY RETRIEVAL SETUP =======================
        print('# ======================= CATEGORY RETRIEVAL SETUP =======================')
        self.category_val_probs = retrieved_cat['val_probs']
        self.category_val_labels = retrieved_cat['val_labels']
        self.category_test_probs = retrieved_cat['test_probs']
        self.category_test_labels = retrieved_cat['test_labels']
        self.category_val_metrics = retrieved_cat['val_metrics']
        self.category_test_metrics = retrieved_cat['test_metrics']

        # ======================= VAL CATEGORY CANDIDATES =======================
        self.category_val_users = []
        self.category_val_candidates = []

        for u, (p, l) in tqdm(enumerate(zip(self.category_val_probs, self.category_val_labels), start=1), desc="VAL Category Candidates"):
            topk = torch.topk(torch.tensor(p), self.args.llm_negative_sample_size + 1).indices.tolist()
            if l in topk:
                self.category_val_users.append(u)
                self.category_val_candidates.append(topk)

        # ======================= TEST CATEGORY CANDIDATES =======================
        self.category_test_users = []
        self.category_test_candidates = []
        self.category_non_test_users = []

        for u, (p, l) in tqdm(enumerate(zip(self.category_test_probs, self.category_test_labels), start=1), desc="TEST Category Candidates"):
            topk = torch.topk(torch.tensor(p), self.args.llm_negative_sample_size + 1).indices.tolist()
            if l in topk:
                self.category_test_users.append(u)
                self.category_test_candidates.append(topk)
            else:
                self.category_non_test_users.append(u)

        # ======================= Build category_test_retrieval dict =======================
        self.category_test_retrieval = {
            'original_size': len(self.category_test_probs),
            'retrieval_size': len(self.category_test_candidates),
            'original_metrics': self.category_test_metrics,
            'retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.category_test_probs)[torch.tensor(self.category_test_users, dtype=torch.long) - 1],
                torch.tensor(self.category_test_labels)[torch.tensor(self.category_test_users, dtype=torch.long) - 1],
                self.args.metric_ks,
            ),
            'non_retrieval_metrics': absolute_recall_mrr_ndcg_for_ks(
                torch.tensor(self.category_test_probs)[torch.tensor(self.category_non_test_users, dtype=torch.long) - 1],
                torch.tensor(self.category_test_labels)[torch.tensor(self.category_non_test_users, dtype=torch.long) - 1],
                self.args.metric_ks,
            ),
        }


            
        
        
    def _generate_10_candidates(self, user_probs, answer_item, answer_category, item_to_category, category_to_items, category_meta_dict):
        """10개 아이템 후보 + 해당 카테고리 후보 5개를 구성"""
        item_candidates = []
        category_candidates = set()
        used_items = set()
        
        # Bounds checking
        if answer_item >= len(user_probs):
            print(f"[WARNING] answer_item {answer_item} is out of user_probs range ({len(user_probs)})")
            return [], []
        
        # 1. 정답 아이템 (반드시 포함)
        item_candidates.append(answer_item)
        used_items.add(answer_item)
        category_candidates.add(answer_category)
        
        # 2. 정답 카테고리에서 추가 아이템 선택 (정답 제외)
        if answer_category in category_to_items:
            items = [i for i in category_to_items[answer_category] 
                    if i not in used_items and i < len(user_probs)]
            if items:
                # 상위 점수 아이템 1개 추가
                best = max(items, key=lambda x: user_probs[x].item())
                item_candidates.append(best)
                used_items.add(best)

        # 3. 다른 카테고리들에서 상위 아이템들 선택
        category_scores = {}
        for cat, items in category_to_items.items():
            if cat == answer_category:
                continue
            valid_items = [i for i in items if i < len(user_probs)]
            if valid_items:
                score = max(user_probs[i].item() for i in valid_items)
                category_scores[cat] = score

        # 상위 4개 카테고리 선택
        top4_cats = sorted(category_scores, key=category_scores.get, reverse=True)[:4]
        category_candidates.update(top4_cats)

        # 각 카테고리에서 상위 2개 아이템 선택
        for cat in top4_cats:
            items = [i for i in category_to_items[cat] 
                    if i not in used_items and i < len(user_probs)]
            if items:
                sorted_items = sorted(items, key=lambda x: user_probs[x].item(), reverse=True)
                added_count = 0
                for i in sorted_items:
                    if len(item_candidates) >= 10:  # 최대 10개로 제한
                        break
                    item_candidates.append(i)
                    used_items.add(i)
                    added_count += 1
                    if added_count >= 2:  # 카테고리당 최대 2개
                        break

        # 4. 아이템 후보가 부족한 경우 전체에서 상위 아이템으로 채우기
        if len(item_candidates) < 10:
            all_items = list(range(len(user_probs)))
            remaining_items = [i for i in all_items if i not in used_items]
            if remaining_items:
                remaining_items.sort(key=lambda x: user_probs[x].item(), reverse=True)
                needed = 10 - len(item_candidates)
                item_candidates.extend(remaining_items[:needed])

        # 5. 카테고리 후보 정리 (최대 5개)
        category_candidates = list(category_candidates)
        # 정답 카테고리를 첫 번째로 배치
        if answer_category in category_candidates:
            category_candidates.remove(answer_category)
        category_candidates = [answer_category] + category_candidates[:4]
        
        # 존재하지 않는 카테고리 제거
        category_candidates = [c for c in category_candidates if c in category_meta_dict]
        
        # 카테고리 후보가 부족한 경우 임의로 채우기
        if len(category_candidates) < 5:
            all_cats = list(category_meta_dict.keys())
            remaining_cats = [c for c in all_cats if c not in category_candidates]
            needed = 5 - len(category_candidates)
            category_candidates.extend(remaining_cats[:needed])

        print(f"[DEBUG] Final item_candidates: {item_candidates[:5]}... (total: {len(item_candidates)})")
        print(f"[DEBUG] Final category_candidates: {category_candidates}")
        print(f"[DEBUG] answer_item {answer_item} in candidates: {answer_item in item_candidates}")

        return item_candidates, category_candidates


    @classmethod
    def code(cls):
        return 'llm'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(
            dataset, 
            batch_size=self.args.lora_micro_batch_size,
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.args.num_workers,
            worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = LLMTrainDataset(
            self.args,
            self.item_train, self.category_train,
            self.args.bert_max_len, self.rng,
            self.item_text_dict, self.category_text_dict,
            self.tokenizer, self.prompter
        )
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        
        # Handle empty dataset case
        if len(dataset) == 0:
            print(f"[WARNING] {mode} dataset is empty, creating dummy dataset")
            # Create a minimal dummy dataset to prevent DataLoader from failing
            dummy_dataset = DummyDataset()
            dataloader = data_utils.DataLoader(dummy_dataset, batch_size=1, shuffle=False,
                                               pin_memory=True, num_workers=0)
            return dataloader
            
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LLMValidDataset(
                self.args,
                self.item_train, self.category_train,
                self.item_val, self.category_val,
                self.args.bert_max_len, self.rng,
                self.item_text_dict, self.category_text_dict,
                self.tokenizer, self.prompter,
                self.item_val_users, self.category_val_users,
                self.item_val_candidates, self.category_val_candidates
            )
            print(f"[DEBUG] Eval dataset length: {len(dataset)}")
        elif mode == 'test':
            dataset = LLMTestDataset(
                self.args,
                self.item_train, self.category_train,
                self.item_val, self.category_val,
                self.item_test, self.category_test,
                self.args.bert_max_len, self.rng,
                self.item_text_dict, self.category_text_dict,
                self.tokenizer, self.prompter,
                self.item_test_users, self.category_test_users,
                self.item_test_candidates, self.category_test_candidates
            )
        return dataset



class LLMTrainDataset(data_utils.Dataset):
    def __init__(self, args, item_u2seq, category_u2seq, max_len, rng, text_dict, category_meta_dict, tokenizer, prompter):
        self.args = args
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.category_meta_dict = category_meta_dict
        self.tokenizer = tokenizer
        self.prompter = prompter

        self.item_to_category = args.item_to_category  # 필수

        self.num_items = args.num_items
        self.num_categories = args.num_categories

        self.all_seqs = []
        for u in sorted(item_u2seq.keys()):
            item_seq = item_u2seq[u]
            category_seq = category_u2seq[u]
            for i in range(2, len(item_seq)+1):
                self.all_seqs.append({
                    'item_seq': item_seq[:i],
                    'category_seq': category_seq[:i]
                })

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        item_tokens = self.all_seqs[index]['item_seq']
        cat_tokens = self.all_seqs[index]['category_seq']

        answer_item = item_tokens[-1]
        answer_cat = cat_tokens[-1]
        item_seq = item_tokens[:-1][-self.max_len:]
        category_seq = cat_tokens[:-1][-self.max_len:]

        # === 후보 아이템: 정답 + 샘플링 9개
        item_candidates = [answer_item]
        cur_idx = 0
        samples = self.rng.randint(1, self.num_items + 1, size=50)
        while len(item_candidates) < 10:
            cand = samples[cur_idx]
            cur_idx += 1
            if cand in item_candidates or cand in item_seq:
                continue
            item_candidates.append(cand)
        self.rng.shuffle(item_candidates)

        # === 후보 카테고리: item 후보에서 추출 후 중복 제거 + 정답 포함 보장
        cat_candidates = []
        for item in item_candidates:
            cat = self.item_to_category.get(item)
            if cat is not None:
                cat_candidates.append(cat)
        cat_candidates = list(set(cat_candidates))
        if answer_cat not in cat_candidates:
            cat_candidates.append(answer_cat)
        while len(cat_candidates) < 5:
            fake_cat = self.rng.randint(1, self.num_categories + 1)
            if fake_cat not in cat_candidates:
                cat_candidates.append(fake_cat)
        self.rng.shuffle(cat_candidates)
        cat_candidates = cat_candidates[:5]

        return seq_to_token_ids(
            self.args, 
            item_seq, 
            item_candidates, 
            answer_item, 
            self.text_dict,
            self.tokenizer,
            self.prompter,
            eval=False,
            category_seq=category_seq,
            category_candidates=cat_candidates,
            category_meta_dict=self.category_meta_dict
        )


        
class LLMValidDataset(data_utils.Dataset):
    def __init__(self, args,
                 item_u2seq, category_u2seq,
                 item_u2answer, category_u2answer,
                 max_len, rng, text_dict, category_meta_dict,
                 tokenizer, prompter,
                 val_users, category_val_users,
                 val_candidates, category_val_candidates):
        self.args = args
        self.item_u2seq = item_u2seq
        self.category_u2seq = category_u2seq
        self.item_u2answer = item_u2answer
        self.category_u2answer = category_u2answer
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.category_meta_dict = category_meta_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.val_users = val_users
        self.category_val_users = category_val_users
        self.val_candidates = val_candidates
        self.category_val_candidates = category_val_candidates

    def __len__(self):
        return len(self.val_users)

    def __getitem__(self, index):
        user = self.val_users[index]
        item_seq = self.item_u2seq[user][-self.max_len:]
        category_seq = self.category_u2seq[user][-self.max_len:]

        item_candidates = self.val_candidates[index]
        category_candidates = self.category_val_candidates[index]

        item_answer = self.item_u2answer[user][0]
        assert item_answer in item_candidates

        return seq_to_token_ids(
            self.args,
            item_seq,
            item_candidates,
            item_answer,
            self.text_dict,
            self.tokenizer,
            self.prompter,
            eval=True,
            category_seq=category_seq,
            category_candidates=category_candidates,
            category_meta_dict=self.category_meta_dict
        )


class LLMTestDataset(data_utils.Dataset):
    def __init__(self, args,
                 item_u2seq, category_u2seq,
                 item_u2val, category_u2val,
                 item_u2answer, category_u2answer,
                 max_len, rng, text_dict, category_meta_dict,
                 tokenizer, prompter,
                 test_users, category_test_users,
                 test_candidates, category_test_candidates):
        self.args = args
        self.item_u2seq = item_u2seq
        self.category_u2seq = category_u2seq
        self.item_u2val = item_u2val
        self.category_u2val = category_u2val
        self.item_u2answer = item_u2answer
        self.category_u2answer = category_u2answer
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.category_meta_dict = category_meta_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.test_users = test_users
        self.category_test_users = category_test_users
        self.test_candidates = test_candidates
        self.category_test_candidates = category_test_candidates

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, index):
        user = self.test_users[index]
        item_seq = (self.item_u2seq[user] + self.item_u2val[user])[-self.max_len:]
        category_seq = (self.category_u2seq[user] + self.category_u2val[user])[-self.max_len:]

        item_candidates = self.test_candidates[index]
        category_candidates = self.category_test_candidates[index]

        item_answer = self.item_u2answer[user][0]
        assert item_answer in item_candidates

        return seq_to_token_ids(
            self.args,
            item_seq,
            item_candidates,
            item_answer,
            self.text_dict,
            self.tokenizer,
            self.prompter,
            eval=True,
            category_seq=category_seq,
            category_candidates=category_candidates,
            category_meta_dict=self.category_meta_dict
        )
