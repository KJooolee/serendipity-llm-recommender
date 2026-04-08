import csv

input_file = 'v2_stats_grouped.csv'

unique_1depth = set()

with open(input_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        unique_1depth.add(row['1-depth'])

# 정렬 후 출력
print(f"1-depth 고유 개수: {len(unique_1depth)}\n")
print("고유 1-depth 목록:")
for cat in sorted(unique_1depth):
    print(f"- {cat}")
