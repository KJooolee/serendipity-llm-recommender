# projector_loader.py
import torch
from model.mlp_projector import MlpProjector  # 경로 맞게 조정

def load_projector(path, rec_size=64, llm_size=4096, device='cpu'):
    model = MlpProjector(rec_size, llm_size)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model