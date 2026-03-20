import json
import random
from collections import defaultdict
from pathlib import Path

# ===== 경로 설정 =====
input_json_path = Path("./datasets/toys_please_Ndiversity.json")  # line JSON 파일
train_out = Path("./datasets/Toys_N/train.txt")
val_out   = Path("./datasets/Toys_N/val.txt")
test_out  = Path("./datasets/Toys_N/test.txt")
item_cat_out = Path("./datasets/Toys_N/item_category.txt")

random.seed(42)  # 재현성을 위해 시드 고정

# ===== 1. JSON 라인 읽기 =====
user_map = {}
item_map = {}
category_map = {}
item_to_category = {}

next_uid = 1
next_iid = 0
next_cid = 0

# uid → [(timestamp, iid, cid)]
user_interactions = defaultdict(list)

with input_json_path.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        uid_raw = obj["uid"]
        sid_raw = obj["sid"]
        cid_raw = obj["category_label"]
        ts      = obj["timestamp"]  # 그대로 사용

        # === user 매핑 ===
        if uid_raw not in user_map:
            user_map[uid_raw] = next_uid
            next_uid += 1
        uid = user_map[uid_raw]

        # === item 매핑 ===
        if sid_raw not in item_map:
            item_map[sid_raw] = next_iid
            next_iid += 1
        iid = item_map[sid_raw]

        # === category 매핑 ===
        if cid_raw not in category_map:
            category_map[cid_raw] = next_cid
            next_cid += 1
        cid = category_map[cid_raw]

        # === item→category 매핑 저장 ===
        item_to_category[iid] = cid

        # === 유저별 인터랙션 기록 (timestamp 포함) ===
        user_interactions[uid].append((ts, iid, cid))

# ===== 2. 유저별 timestamp 정렬 & [-1] test, [-2] val, 그 외 train =====
train_pairs = []
val_pairs = []
test_pairs = []

for uid in sorted(user_interactions.keys()):
    interactions = sorted(user_interactions[uid], key=lambda x: x[0])  # ts 오름차순
    n = len(interactions)
    if n == 0:
        continue
    if n == 1:
        test_pairs.append((uid, interactions[-1][1]))
        continue
    if n == 2:
        val_pairs.append((uid, interactions[-2][1]))
        test_pairs.append((uid, interactions[-1][1]))
        continue

    test_pairs.append((uid, interactions[-1][1]))   # 마지막 → test
    val_pairs.append((uid, interactions[-2][1]))    # 마지막-1 → val
    for _, iid, _ in interactions[:-2]:             # 나머지 → train
        train_pairs.append((uid, iid))

# ===== 3. 저장 =====
def save_pairs(pairs, path):
    with path.open('w', encoding='utf-8') as f:
        for uid, iid in pairs:
            f.write(f"{uid},{iid}\n")

save_pairs(train_pairs, train_out)
save_pairs(val_pairs, val_out)
save_pairs(test_pairs, test_out)

with item_cat_out.open('w', encoding='utf-8') as f:
    for iid, cid in sorted(item_to_category.items()):
        f.write(f"{iid},{cid}\n")

print(f"train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")
print(f"users: {len(user_map)}, items: {len(item_map)}, categories: {len(category_map)}")
