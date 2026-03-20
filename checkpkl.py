import pickle

def inspect_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    val_probs = data['val_probs']
    val_labels = data['val_labels']
    print(f"[INFO] val_probs length: {len(val_probs)}")
    print(f"[INFO] val_labels length: {len(val_labels)}")
    print("\n[Sample val_probs]:", val_probs[:2])
    print("[Sample val_labels]:", val_labels[:2])
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)

    # print(f"\n[INFO] Inspecting: {path}")
    # print(f"- Keys: {list(data.keys())}")

    # print("\n[DEBUG] smap (sid → item_idx):")
    # for i, (k, v) in enumerate(data['smap'].items()):
    #     print(f"  {i+1}. {k} → {v}")
    #     if i == 4:
    #         break

    # print("\n[DEBUG] umap (uid → user_idx):")
    # for i, (k, v) in enumerate(data['umap'].items()):
    #     print(f"  {i+1}. {k} → {v}")
    #     if i == 4:
    #         break
    # meta = data['meta'] 
    # print("\n[DEBUG] meta (item_idx → metadata keys):")
    # for i, (k, v) in enumerate(list(meta.items())[:5]):
    #     if isinstance(v, dict):
    #         print(f"  {i+1}. {k} → {list(v.keys())}")
    #     else:
    #         print(f"  {i+1}. {k} → {v}")
    #     if i == 4:
    #         break
    # print("\n[DEBUG] meta['asin'] (asin → item_idx):")
    # for i, (asin, idx) in enumerate(list(meta['asin'].items())[:5]):
    #     print(f"  {i+1}. {asin} → {idx}")    
            
# with open('./data/preprocessed/home_category_min_rating0-min_uc5-min_sc5/dataset.pkl', 'rb') as f:
#     data = pickle.load(f)

#meta = data['meta']
#print(meta)  # 샘플 키
    
# 사용 예시:
# inspect_pkl("./data/preprocessed/home_category_min_rating0-min_uc5-min_sc5/dataset.pkl")
# inspect_pkl("./data/preprocessed/home_min_rating0-min_uc5-min_sc5/dataset.pkl")
# inspect_pkl("/home/work/sg/LlamaRec-doit/experiments/lru/home/retrieved.pkl")
# import json

# path = 'data/home/home_category_meta.json'

# with open(path, 'r') as f:
#     data = json.load(f)

# # asin 중복 없이 개수 세기
# asins = set()
# for item in data:
#     if 'asin' in item:
#         asins.add(item['asin'])

# print(f"중복 없는 asin 개수: {len(asins)}")
import pickle

def check_dataset_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n✅ [CHECKING] {path}")
    print("전체 키 목록:", list(data.keys()))
    
    for key in ['train', 'val', 'test']:
        if key in data:
            sample_keys = list(data[key].keys())[:3]
            print(f"  └─ {key}: {len(data[key])} users, 예시: {{uid: seq}} →")
            for uid in sample_keys:
                print(f"     {uid}: {data[key][uid]}")
    
    if 'meta' in data:
        print(f"  └─ meta: {len(data['meta'])} entries")
        example = list(data['meta'].items())[:3]
        for k, v in example:
            print(f"     {k}: {v}")
    
    if 'umap' in data:
        print(f"  └─ umap: {len(data['umap'])} users, 예시: {list(data['umap'].items())[:3]}")
    if 'smap' in data:
        print(f"  └─ smap: {len(data['smap'])} items, 예시: {list(data['smap'].items())[:3]}")
    if 'cmap' in data:
        print(f"  └─ cmap: {len(data['cmap'])} categories, 예시: {list(data['cmap'].items())[:3]}")
        
check_dataset_pkl("./preprocessed/home_category_min_rating0-min_uc5-min_sc5/dataset.pkl")
check_dataset_pkl("./preprocessed/home_min_rating0-min_uc5-min_sc5/dataset.pkl")