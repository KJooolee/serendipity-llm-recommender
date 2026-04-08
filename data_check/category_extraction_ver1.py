import csv
import ast
from collections import defaultdict
from tqdm import tqdm

input_file = 'metadata.json'  # 한 줄당 Python dict 문자열
output_file = 'v1.csv'

category_paths = defaultdict(list)
category_validity = defaultdict(list)

# 총 줄 수 계산
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 본 처리
with open(input_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(tqdm(f, total=total_lines, desc="Processing items")):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue

        try:
            item = ast.literal_eval(line)
            categories = item.get("categories", [])

            # 핵심 조건: 리스트 안에 리스트가 1개만 있어야 한다
            if not isinstance(categories, list) or len(categories) != 1:
                continue

            category_list = categories[0]
            if not isinstance(category_list, list) or len(category_list) == 0:
                continue

            top_cat = category_list[0]
            if not top_cat:
                continue

            category_paths[top_cat].append(category_list)
            category_validity[top_cat].append(True)

        except Exception as e:
            tqdm.write(f"[에러] {e}")
            continue

# 유효한 1-depth만 필터링
valid_top_categories = {
    k for k, v in category_validity.items()
    if all(v)
}

# CSV 저장 (엑셀 호환)
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['1-depth', 'category_path'])
    for top_cat in valid_top_categories:
        for path in category_paths[top_cat]:
            writer.writerow([top_cat, str(path)])
