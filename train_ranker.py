# export TRANSFORMERS_CACHE=/home/work/sg/.hf_cache
# export HF_HOME=/home/work/sg/.hf_cache
import importlib
hf_datasets = importlib.import_module("datasets")  # HuggingFace datasets 강제 import

import transformers.trainer
transformers.trainer.datasets = hf_datasets  # Trainer 내부에서 사용하는 datasets 참조 우회


import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import copy
import argparse
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *
from model.mlp_projector import MlpProjector
from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from model import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)
import torch.multiprocessing as mp
from optimizer import LinearWarmupCosineLRScheduler
mp.set_start_method('spawn', force=True)
from trainer.llm import llama_collate_fn

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')
    


    
##
def load_or_train_projector(path, rec_size=64, llm_size=4096, device='cuda'):
    projector = MlpProjector(rec_size=rec_size, llm_size=llm_size).to(device)

    if os.path.exists(path):
        print(f"[INFO] Loading projector from {path}")
        state_dict = torch.load(path, map_location=device)
        projector.load_state_dict(state_dict)
    else:
        print(f"[INFO] Projector not found at {path}, training new one...")

        train_dummy_projector(projector, device)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(projector.state_dict(), path)
        print(f"[INFO] Projector saved to {path}")
    
    return projector


def train_dummy_projector(projector, device='cuda'):
    projector.train()
    optimizer = torch.optim.Adam(projector.parameters(), lr=1e-3)

    dummy_input = torch.randn(128, 64).to(device)
    dummy_target = torch.randn(128, 4096).to(device)

    for epoch in range(5):
        optimizer.zero_grad()
        pred = projector(dummy_input)
        loss = torch.nn.functional.mse_loss(pred, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"[Projector Train] Epoch {epoch} - Loss: {loss.item():.4f}")

def main(args, export_root=None):
    args.llm_base_tokenizer = 'meta-llama/Llama-2-7b-hf'
    seed_everything(args.seed)
    if export_root is None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    # ① 먼저 LRURec 모델 정의
    # item용 args 복사 및 모델 정의
    args_item = copy.deepcopy(args)
    args_item.num_items = 759  #1011#19925 ###41062 
    rec_model_item = LRURec(args_item)
    item_checkpoint = torch.load(args.item_model_path, map_location="cpu")
    rec_model_item.load_state_dict(item_checkpoint["model_state_dict"])
    rec_model_item.eval().to(args.device)
    for p in rec_model_item.parameters():
        p.requires_grad = False

    # category용 args 복사 및 모델 정의
    args_category = copy.deepcopy(args)
    args_category.num_items = 52 #83 #89
    rec_model_category = LRURec(args_category)
    category_checkpoint = torch.load(args.category_model_path, map_location="cpu")
    rec_model_category.load_state_dict(category_checkpoint["model_state_dict"])
    rec_model_category.eval().to(args.device)

    for p in rec_model_category.parameters():
        p.requires_grad = False
 
    train_loader, val_loader, test_loader, tokenizer, test_retrieval, dataloader = dataloader_factory(
        args,
        rec_model_item=rec_model_item,
        rec_model_category=rec_model_category
    )

    print("[DEBUG] tokenizer:", tokenizer)
    print("[DEBUG] tokenizer type:", type(tokenizer))
    # 3. 저장


    # ③ projector 정의
    item_projector = load_or_train_projector(
        args.item_projector_pth_path,
        rec_size=64,
        llm_size=4096,
        device=args.device
    )
    category_projector = load_or_train_projector(
        args.category_projector_pth_path,
        rec_size=64,
        llm_size=4096,
        device=args.device
    )

    # ④ LLM 모델 로딩 및 PEFT
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.llm_cache_dir,
    )
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[HistoryEmb]', '[CansEmb]', '[ItemEmb]']
    })
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model = LLMWithProjector(
        base_model=model,  # get_peft_model로 LoRA 적용된 huggingface model
        item_projector=item_projector,
        category_projector=category_projector,
        rec_model_item=rec_model_item,
        rec_model_category=rec_model_category,
        tokenizer=tokenizer
    )
    print("[DEBUG] Projector requires_grad:")
    for name, param in model.named_parameters():
        if "projector" in name:
            print(f"{name} → requires_grad={param.requires_grad}")
    print("[DEBUG] model wrapper:", model.__class__)
    # ⑤ trainer 생성 및 학습
    trainer = LLMTrainer(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
        export_root,
        args.use_wandb,
        #item_projector=item_projector,
        #category_projector=category_projector,
        rec_model_item=rec_model_item,
        rec_model_category=rec_model_category
    )
    print("[DEBUG] Projector requires_grad:")
    for name, param in model.named_parameters():
        if "projector" in name:
            print(f"{name} → requires_grad={param.requires_grad}")
    trainer.train()
    trainer.test(test_retrieval)

if __name__ == "__main__":
    args.model_code = 'llm'
    set_template(args)
    main(args, export_root=None)
