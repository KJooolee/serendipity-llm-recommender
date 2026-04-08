import json
import csv

input_path = 'metadata.json'
output_path = 'categories_only.csv'

with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8', newline='') as fout:
    
    writer = csv.writer(fout)
    writer.writerow(['categories'])  # 헤더

    for line in fin:
        try:
            item = json.loads(line)
            categories = item.get('categories', [])
            writer.writerow([categories])  # 리스트 그대로 저장
        except json.JSONDecodeError:
            continue  # JSON 파싱 실패한 줄은 무시

print("CSV 저장 완료:", output_path)
