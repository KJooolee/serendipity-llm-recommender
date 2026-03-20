import torch
import torch.nn as nn

class LLMWithProjector(nn.Module):
    def __init__(self, llm, item_projector, category_projector, rec_model_item, rec_model_category, tokenizer):
        super().__init__()
        self.llm = llm
        self.item_projector = item_projector
        self.category_projector = category_projector
        self.rec_model_item = rec_model_item
        self.rec_model_category = rec_model_category
        self.tokenizer = tokenizer

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def encode_items(self, item_ids, projector, rec_model):
        with torch.no_grad():
            rec_embs = rec_model.embedding.token(item_ids)  # [B, N, D_rec]
        proj_embs = projector(rec_embs)  # [B, N, D_llm]
        return proj_embs

    def wrap_emb(self, input_ids, seq_ids, cat_ids):
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, D]

        his_token_id = self.tokenizer("[HistoryEmb]", add_special_tokens=False).input_ids[0]
        cans_token_id = self.tokenizer("[CansEmb]", add_special_tokens=False).input_ids[0]

        his_embs = self.encode_items(seq_ids, self.item_projector, self.rec_model_item).to(input_ids.device)
        cans_embs = self.encode_items(cat_ids, self.category_projector, self.rec_model_category).to(input_ids.device)  # [B, C, D]

        for i in range(input_ids.size(0)):
            # Insert history embeddings
            idxs = (input_ids[i] == his_token_id).nonzero(as_tuple=True)[0]
            for j, idx in enumerate(idxs):
                if j < his_embs.size(1):  # guard
                    input_embeds[i, idx] = his_embs[i, j]

            # Insert candidate category embeddings
            idxs = (input_ids[i] == cans_token_id).nonzero(as_tuple=True)[0]
            for j, idx in enumerate(idxs):
                if j < cans_embs.size(1):
                    input_embeds[i, idx] = cans_embs[i, j]

        return input_embeds

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        item_embs = kwargs["item_embs"]
        cat_embs = kwargs["cat_embs"]

        inputs_embeds = self.wrap_emb(input_ids, item_embs, cat_embs)
        # print(f"[DEBUG] input_ids.shape: {input_ids.shape}")
        # print(f"[DEBUG] attention_mask.shape: {attention_mask.shape}")
        # print(f"[DEBUG] labels type: {type(labels)}")
        if isinstance(labels, torch.Tensor):
            print(f"[DEBUG] labels.shape: {labels.shape}")
        else:
            print(f"[DEBUG] labels content: {labels}")
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
