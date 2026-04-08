import ast
import csv
from tqdm import tqdm

input_file = 'metadata.json'

# 전체 카테고리
target_categories = {
    'Amazon Fashion', 'All Beauty', 'Appliances', 'Arts, Crafts & Sewing', 'Automotive', 'Books', 'Beauty', 'CDs & Vinyl',
    'Cell Phones & Accessories', 'Clothing, Shoes & Jewelry', 'Digital Music', 'Electronics', 'Gift Cards',
    'Grocery & Gourmet Food', 'Health & Personal Care', 'Home & Kitchen', 'Industrial & Scientific', 'Kindle Store',
    'Luxury Beauty', 'Magazine Subscriptions', 'Movies & TV', 'Musical Instruments', 'Office Products',
    'Patio, Lawn & Garden', 'Pet Supplies', 'Prime Pantry', 'Software', 'Sports & Outdoors', 'Tools & Home Improvement',
    'Toys & Games', 'Video Games'
}

# 결과 저장용 dict
results = {cat: {'valid': 0, 'invalid': 0} for cat in target_categories}

# 총 줄 수 계산
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 본 처리
with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=total_lines, desc="검증 중"):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue

        try:
            item = ast.literal_eval(line)
            categories = item.get("categories", [])

            if (
                isinstance(categories, list)
                and len(categories) >= 1
                and isinstance(categories[0], list)
                and len(categories[0]) >= 1
            ):
                first_cat = categories[0][0]
                if first_cat in results:
                    if len(categories) == 1:
                        results[first_cat]['valid'] += 1
                    else:
                        results[first_cat]['invalid'] += 1
        except Exception:
            continue  # 깨진 줄 무시

# CSV로 저장
output_file = '카테고리_단일or다중리스트_여부.csv'
with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Category', 'Single List (Valid)', 'Multi List (Invalid)', 'Total'])
    for cat in sorted(results.keys()):
        v = results[cat]
        writer.writerow([cat, v['valid'], v['invalid'], v['valid'] + v['invalid']])
