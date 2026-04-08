import csv
import ast
from collections import defaultdict

input_file = 'v2.csv'
output_file = 'beauty_electronics_depth3_mixed.csv'

target_categories = {'Beauty', 'Electronics'}
depth_counter = defaultdict(int)

with open(input_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path_str = row['category_path'].strip()

        try:
            category_list = ast.literal_eval(path_str)

            if (
                isinstance(category_list, list)
                and len(category_list) >= 2
                and category_list[0] in target_categories
                and all(x.strip() for x in category_list[:2])
            ):
                d1 = category_list[0]
                d2 = category_list[1]
                d3 = category_list[2] if len(category_list) >= 3 and category_list[2].strip() else ''
                depth_counter[(d1, d2, d3)] += 1

        except Exception as e:
            print(f"[에러] {path_str} 변환 실패: {e}")

# 저장
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['1-depth', '2-depth', '3-depth', 'count'])

    for (d1, d2, d3), count in sorted(
        depth_counter.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][2], -x[1])
    ):
        writer.writerow([d1, d2, d3, f"{count:,}"])
