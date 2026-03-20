from torch.utils.data import Dataset
import torch

class DynamicPromptDataset(Dataset):
    def __init__(self, data_df, rec_model_item, rec_model_category, projector, tokenizer, meta_dict, args):
        self.df = data_df.reset_index(drop=True)
        self.rec_model_item = rec_model_item
        self.rec_model_category = rec_model_category
        self.projector = projector
        self.tokenizer = tokenizer
        self.meta_dict = meta_dict
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']  # 시퀀스 = item 인덱스 리스트
        
        # 🔹 1. 임베딩 뽑기
        with torch.no_grad():
            item_emb = self.rec_model_item.encode_items(torch.tensor(seq).unsqueeze(0).to("cuda"))  # (1, len, dim)
            cat_emb = self.rec_model_category.encode_items(torch.tensor(seq).unsqueeze(0).to("cuda"))

            item_proj = self.projector(item_emb)  # (1, len, 4096)
            cat_proj = self.projector(cat_emb)

        # 🔹 2. 타이틀 및 카테고리 타이틀 추출
        titles = [self.meta_dict.get(str(sid), "Unknown") for sid in seq]
        cat_titles = ["(category)" for _ in seq]  # 카테고리 타이틀 넣을 수 있으면 여기 교체

        # 🔹 3. 프롬프트 생성 (프롬프트 템플릿은 자유롭게 조정 가능)
        history = []
        for title, i_emb, c_emb, c_title in zip(titles, item_proj[0], cat_proj[0], cat_titles):
            # 여긴 예시야, 실제로 임베딩은 안 보이니까 placeholder만 써
            history.append(f"{title} [item_emb] – {c_title} [cat_emb]")

        prompt = f"This user has watched " + ", ".join(history) + ". Recommend next movie:"

        # 🔹 4. 토크나이즈
        tokenized = self.tokenizer(prompt, padding=False, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
