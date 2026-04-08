import pandas as pd

df = pd.read_csv("ratings_Beauty.csv", header=None)

df.columns = ['uid', 'sid', 'rating', 'timestamp']

# 고유한 상품 수 세기
num_unique_items = df['sid'].nunique()

print(f"서로 다른 상품 개수: {num_unique_items}")
