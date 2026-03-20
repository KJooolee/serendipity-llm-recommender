from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import pickle
import numpy as np
from abc import *
from pathlib import Path
import os

def evaluate_final_metrics(export_root, ks=[1, 5, 10, 20]):
    def load(name, part):
        with open(f"{export_root}/{name}_part{part}.pkl", 'rb') as f:
            return pickle.load(f)

    def eval_single(prefix):
        probs1 = load(f"{prefix}_probs", 1)
        labels1 = load(f"{prefix}_labels", 1)
        metrics1 = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(probs1, dtype=torch.float16),
            torch.tensor(labels1),
            ks
        )
        del probs1, labels1
        gc.collect()

        probs2 = load(f"{prefix}_probs", 2)
        labels2 = load(f"{prefix}_labels", 2)
        metrics2 = absolute_recall_mrr_ndcg_for_ks(
            torch.tensor(probs2, dtype=torch.float16),
            torch.tensor(labels2),
            ks
        )
        del probs2, labels2
        gc.collect()

        return {k: (metrics1[k] + metrics2[k]) / 2 for k in ks}

    val_metrics = eval_single("val")
    test_metrics = eval_single("test")

    with open(f"{export_root}/val_metrics.pkl", 'wb') as f:
        pickle.dump(val_metrics, f)
    with open(f"{export_root}/test_metrics.pkl", 'wb') as f:
        pickle.dump(test_metrics, f)

    print("[VAL METRICS]", val_metrics)
    print("[TEST METRICS]", test_metrics)
    
class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
    
    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        seqs, labels = batch
        
        scores = self.model(seqs)[:, -1, :]
        B, L = seqs.shape
        if exclude_history:
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics
    def generate_candidates(self, retrieved_data_path):
        self.model.eval()
        val_probs, val_labels = [], []
        test_probs, test_labels = [], []
        with torch.no_grad():
            print('*************** Generating Candidates for Validation Set ***************')
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                val_probs.extend(scores.tolist())
                val_labels.extend(labels.view(-1).tolist())
            val_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(val_probs), 
                                                          torch.tensor(val_labels).view(-1), self.metric_ks)
            print(val_metrics)

            print('****************** Generating Candidates for Test Set ******************')
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, labels = batch
        
                scores = self.model(seqs)[:, -1, :]
                B, L = seqs.shape
                for i in range(L):
                    scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
                scores[:, 0] = -1e9  # padding
                test_probs.extend(scores.tolist())
                test_labels.extend(labels.view(-1).tolist())
            test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
                                                           torch.tensor(test_labels).view(-1), self.metric_ks)
            print(test_metrics)

        with open(retrieved_data_path, 'wb') as f:
            pickle.dump({'val_probs': val_probs,
                         'val_labels': val_labels,
                         'val_metrics': val_metrics,
                         'test_probs': test_probs,
                         'test_labels': test_labels,
                         'test_metrics': test_metrics}, f)    
    # def generate_candidates(self, retrieved_data_path):
    #     self.model.eval()
        
    #     # 배치 단위로 저장하도록 개선
    #     batch_size_limit = 1000  # 메모리에 유지할 최대 샘플 수
        
    #     val_probs, val_labels = [], []
    #     test_probs, test_labels = [], []
        
    #     with torch.no_grad():
    #         print('*************** Generating Candidates for Validation Set ***************')
    #         tqdm_dataloader = tqdm(self.val_loader)
            
    #         for batch_idx, batch in enumerate(tqdm_dataloader):
    #             batch = self.to_device(batch)
    #             seqs, labels = batch
        
    #             scores = self.model(seqs)[:, -1, :]
    #             B, L = seqs.shape
    #             for i in range(L):
    #                 scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    #             scores[:, 0] = -1e9  # padding
                
    #             # CPU로 이동하고 즉시 정리
    #             scores_cpu = scores.cpu()
    #             labels_cpu = labels.view(-1).cpu()
                
    #             val_probs.extend(scores_cpu.tolist())
    #             val_labels.extend(labels_cpu.tolist())
                
    #             # 메모리 정리
    #             del scores, scores_cpu, labels_cpu
    #             torch.cuda.empty_cache()
                
    #             # 주기적으로 중간 저장 (메모리 한계 방지)
    #             if len(val_probs) >= batch_size_limit:
    #                 # 임시 저장 후 메모리 비우기
    #                 temp_data = {
    #                     'val_probs_batch': val_probs.copy(),
    #                     'val_labels_batch': val_labels.copy()
    #                 }
                    
    #                 # 기존 데이터와 병합하여 저장
    #                 temp_path = retrieved_data_path.replace('.pkl', f'_temp_val_{batch_idx}.pkl')
    #                 with open(temp_path, 'wb') as f:
    #                     pickle.dump(temp_data, f)
                    
    #                 # 메모리에서 제거
    #                 val_probs.clear()
    #                 val_labels.clear()
                    
    #                 import gc
    #                 gc.collect()
            
    #         # 최종 검증 메트릭 계산
    #         val_metrics = self._calculate_final_metrics_from_files(
    #             retrieved_data_path, 'val', self.metric_ks)
    #         print(val_metrics)

    #         print('****************** Generating Candidates for Test Set ******************')
    #         tqdm_dataloader = tqdm(self.test_loader)
            
    #         for batch_idx, batch in enumerate(tqdm_dataloader):
    #             batch = self.to_device(batch)
    #             seqs, labels = batch
        
    #             scores = self.model(seqs)[:, -1, :]
    #             B, L = seqs.shape
    #             for i in range(L):
    #                 scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    #             scores[:, 0] = -1e9  # padding
                
    #             # CPU로 이동하고 즉시 정리
    #             scores_cpu = scores.cpu()
    #             labels_cpu = labels.view(-1).cpu()
                
    #             test_probs.extend(scores_cpu.tolist())
    #             test_labels.extend(labels_cpu.tolist())
                
    #             # 메모리 정리
    #             del scores, scores_cpu, labels_cpu
    #             torch.cuda.empty_cache()
                
    #             # 주기적으로 중간 저장
    #             if len(test_probs) >= batch_size_limit:
    #                 temp_data = {
    #                     'test_probs_batch': test_probs.copy(),
    #                     'test_labels_batch': test_labels.copy()
    #                 }
                    
    #                 temp_path = retrieved_data_path.replace('.pkl', f'_temp_test_{batch_idx}.pkl')
    #                 with open(temp_path, 'wb') as f:
    #                     pickle.dump(temp_data, f)
                    
    #                 test_probs.clear()
    #                 test_labels.clear()
                    
    #                 import gc
    #                 gc.collect()
            
    #         # 최종 테스트 메트릭 계산
    #         test_metrics = self._calculate_final_metrics_from_files(
    #             retrieved_data_path, 'test', self.metric_ks)
    #         print(test_metrics)

    #     # 모든 임시 파일들을 하나로 병합하여 최종 저장
    #     self._merge_temp_files_and_save(retrieved_data_path, val_metrics, test_metrics)


    # def _calculate_final_metrics_from_files(self, base_path, prefix, metric_ks):
    #     """임시 파일들로부터 최종 메트릭 계산"""
    #     import glob
        
    #     all_probs = []
    #     all_labels = []
        
    #     # 임시 파일들 찾기
    #     temp_files = glob.glob(base_path.replace('.pkl', f'_temp_{prefix}_*.pkl'))
        
    #     for temp_file in temp_files:
    #         with open(temp_file, 'rb') as f:
    #             temp_data = pickle.load(f)
    #             all_probs.extend(temp_data[f'{prefix}_probs_batch'])
    #             all_labels.extend(temp_data[f'{prefix}_labels_batch'])
        
    #     if all_probs:
    #         metrics = absolute_recall_mrr_ndcg_for_ks(
    #             torch.tensor(all_probs), 
    #             torch.tensor(all_labels).view(-1), 
    #             metric_ks
    #         )
    #     else:
    #         metrics = {}
        
    #     return metrics


    # def _merge_temp_files_and_save(self, retrieved_data_path, val_metrics, test_metrics):
    #     """임시 파일들을 병합하여 최종 파일로 저장"""
    #     import glob
    #     import os
        
    #     final_data = {
    #         'val_probs': [],
    #         'val_labels': [],
    #         'val_metrics': val_metrics,
    #         'test_probs': [],
    #         'test_labels': [],
    #         'test_metrics': test_metrics
    #     }
        
    #     # validation 임시 파일들 병합
    #     val_temp_files = glob.glob(retrieved_data_path.replace('.pkl', '_temp_val_*.pkl'))
    #     for temp_file in val_temp_files:
    #         with open(temp_file, 'rb') as f:
    #             temp_data = pickle.load(f)
    #             final_data['val_probs'].extend(temp_data['val_probs_batch'])
    #             final_data['val_labels'].extend(temp_data['val_labels_batch'])
    #         os.remove(temp_file)  # 임시 파일 삭제
        
    #     # test 임시 파일들 병합
    #     test_temp_files = glob.glob(retrieved_data_path.replace('.pkl', '_temp_test_*.pkl'))
    #     for temp_file in test_temp_files:
    #         with open(temp_file, 'rb') as f:
    #             temp_data = pickle.load(f)
    #             final_data['test_probs'].extend(temp_data['test_probs_batch'])
    #             final_data['test_labels'].extend(temp_data['test_labels_batch'])
    #         os.remove(temp_file)  # 임시 파일 삭제
        
    #     # 최종 저장
    #     with open(retrieved_data_path, 'wb') as f:
    #         pickle.dump(final_data, f)    
            
            
    # def generate_candidates(self, retrieved_data_path):
    #     self.model.eval()
    #     val_probs, val_labels = [], []
    #     test_probs, test_labels = [], []
    #     with torch.no_grad():
    #         print('*************** Generating Candidates for Validation Set ***************')
    #         tqdm_dataloader = tqdm(self.val_loader)
    #         for batch_idx, batch in enumerate(tqdm_dataloader):
    #             batch = self.to_device(batch)
    #             seqs, labels = batch
        
    #             scores = self.model(seqs)[:, -1, :]
    #             B, L = seqs.shape
    #             for i in range(L):
    #                 scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    #             scores[:, 0] = -1e9  # padding
    #             val_probs.extend(scores.tolist())
    #             val_labels.extend(labels.view(-1).tolist())
    #         val_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(val_probs), 
    #                                                       torch.tensor(val_labels).view(-1), self.metric_ks)
    #         print(val_metrics)

    #         print('****************** Generating Candidates for Test Set ******************')
    #         tqdm_dataloader = tqdm(self.test_loader)
    #         for batch_idx, batch in enumerate(tqdm_dataloader):
    #             batch = self.to_device(batch)
    #             seqs, labels = batch
        
    #             scores = self.model(seqs)[:, -1, :]
    #             B, L = seqs.shape
    #             for i in range(L):
    #                 scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    #             scores[:, 0] = -1e9  # padding
    #             test_probs.extend(scores.tolist())
    #             test_labels.extend(labels.view(-1).tolist())
    #         test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
    #                                                        torch.tensor(test_labels).view(-1), self.metric_ks)
    #         print(test_metrics)

    #     with open(retrieved_data_path, 'wb') as f:
    #         pickle.dump({'val_probs': val_probs,
    #                      'val_labels': val_labels,
    #                      'val_metrics': val_metrics,
    #                      'test_probs': test_probs,
    #                      'test_labels': test_labels,
    #                      'test_metrics': test_metrics}, f)    
