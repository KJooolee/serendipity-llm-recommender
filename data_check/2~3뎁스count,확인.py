import ast
import csv
from collections import defaultdict
from tqdm import tqdm

input_file = 'metadata.json'
target_1depth = {"Electronics", "Home & Kitchen"}

# (top_cat, depth_level, parent_category, sub_category) → count
depth_entries = defaultdict(int)

# 줄 수 계산
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 본 처리
with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc="2, 3-depth 추출 중"):
        try:
            item = ast.literal_eval(line)
            categories = item.get("categories", [])
            if not categories or not isinstance(categories[0], list):
                continue

            path = categories[0]
            if not path or path[0] not in target_1depth:
                continue

            top_cat = path[0]
            if len(path) >= 2:
                # 2-depth
                depth_entries[(top_cat, 2, path[0], path[1])] += 1
            if len(path) >= 3:
                # 3-depth
                depth_entries[(top_cat, 3, path[1], path[2])] += 1
        except Exception:
            continue

# CSV 저장
output_file = '2_3뎁스확인.csv'
with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['depth_level', 'depth_1_category', 'depth_2_category', 'depth_3_category', 'count'])

    for (top_cat, depth_level, parent, sub), count in sorted(depth_entries.items()):
        if depth_level == 2:
            writer.writerow([2, top_cat, sub, '', count])
        elif depth_level == 3:
            writer.writerow([3, top_cat, parent, sub, count])
