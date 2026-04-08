import csv
import ast
from tqdm import tqdm

input_file = 'metadata.json'   # 한 줄당 Python dict 문자열
output_file = 'v2.csv'

# 총 줄 수 미리 계산
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

valid_categories = []

# 본 처리
with open(input_file, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(tqdm(f, total=total_lines, desc="Filtering categories")):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue

        try:
            item = ast.literal_eval(line)
            categories = item.get("categories", [])

            # 리스트 안에 리스트가 하나만 있어야 함
            if isinstance(categories, list) and len(categories) == 1:
                category_list = categories[0]
                if isinstance(category_list, list) and len(category_list) > 0:
                    valid_categories.append(category_list)

        except Exception as e:
            tqdm.write(f"[에러] {e}")
            continue

# CSV로 저장 (엑셀 호환)
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['category_path'])  # 헤더: 단일 컬럼
    for path in valid_categories:
        writer.writerow([str(path)])     # 리스트 문자열로 저장
