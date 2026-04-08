# import csv
# import ast
# from collections import defaultdict

# input_file = 'v2.csv'
# output_file = 'v2_stats.csv'
# depth2_counter = defaultdict(int)

# with open(input_file, 'r', encoding='utf-8-sig') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         path_str = row['category_path'].strip()

#         try:
#             # 문자열을 리스트로 안전하게 변환
#             category_list = ast.literal_eval(path_str)

#             # 공백 리스트 등 필터링
#             if (
#                 isinstance(category_list, list)
#                 and len(category_list) >= 2
#                 and all(x.strip() for x in category_list[:2])  # 앞 2개가 비어있지 않은 경우
#             ):
#                 d1, d2 = category_list[0], category_list[1]
#                 depth2_counter[(d1, d2)] += 1

#         except Exception as e:
#             print(f"[에러] {path_str} 변환 실패: {e}")

# # 저장
# with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
#     writer = csv.writer(f)
#     writer.writerow(['1-depth', '2-depth', 'count'])
#     for (d1, d2), count in sorted(depth2_counter.items(), key=lambda x: -x[1]):
#         writer.writerow([d1, d2, count])

import csv
import ast
from collections import defaultdict

input_file = 'v2.csv'
output_file = 'v2_stats_grouped.csv'

depth2_counter = defaultdict(int)
depth1_total = defaultdict(int)

with open(input_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path_str = row['category_path'].strip()

        try:
            category_list = ast.literal_eval(path_str)

            if (
                isinstance(category_list, list)
                and len(category_list) >= 2
                and all(x.strip() for x in category_list[:2])
            ):
                d1, d2 = category_list[0], category_list[1]
                depth2_counter[(d1, d2)] += 1
                depth1_total[d1] += 1

        except Exception as e:
            print(f"[에러] {path_str} 변환 실패: {e}")

# 저장
with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['1-depth', '2-depth', 'count', '1-depth total'])

    for (d1, d2), count in sorted(depth2_counter.items(), key=lambda x: (x[0], -x[1])):
        writer.writerow([d1, d2, f"{count:,}", f"{depth1_total[d1]:,}"])
