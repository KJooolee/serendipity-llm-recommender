import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

# 파일 경로 지정
train_p = Path("./datasets/Beauty/train.txt")
val_p   = Path("./datasets/Beauty/val.txt")
test_p  = Path("./datasets/Beauty/test.txt")

def load(fp):
    # 형식: "user_id,item_id" 헤더 없음 가정
    df = pd.read_csv(fp, header=None, names=["user","item"])
    # 공백 제거 & 정수 변환 시도
    df["user"] = df["user"].astype(str).str.strip()
    df["item"] = df["item"].astype(str).str.strip()
    return df

tr = load(train_p)
va = load(val_p)
te = load(test_p)

# 0) 기본 통계
print("Rows  | train/val/test:", len(tr), len(va), len(te))
print("Users | train/val/test:", tr.user.nunique(), va.user.nunique(), te.user.nunique())
print("Items | train/val/test:", tr.item.nunique(), va.item.nunique(), te.item.nunique())

# 1) (user,item) 중복/누수 점검
dupes = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]) \
          .duplicated(subset=["user","item"], keep=False)
print("Any exact (user,item) duplicates across splits? ->", bool(dupes.any()))

# 2) 유저 커버리지
u_tr, u_va, u_te = set(tr.user), set(va.user), set(te.user)
print("Users only in train:", len(u_tr - u_va - u_te))
print("Users in all splits:", len(u_tr & u_va & u_te))

# 3) per-user 분할 비율
def per_user_counts(df): return df.groupby("user").size()
ct_tr, ct_va, ct_te = per_user_counts(tr), per_user_counts(va), per_user_counts(te)
users_all = sorted(set(ct_tr.index) | set(ct_va.index) | set(ct_te.index))

records = []
for u in users_all:
    a = ct_tr.get(u, 0); b = ct_va.get(u, 0); c = ct_te.get(u, 0)
    tot = a + b + c
    if tot == 0: continue
    records.append((u, a, b, c, a/tot, b/tot, c/tot, tot))
df_ratio = pd.DataFrame(records, columns=["user","n_train","n_val","n_test","r_train","r_val","r_test","n_total"])

print("\nPer-user ratio summary (mean ± std):")
for col in ["r_train","r_val","r_test"]:
    print(f"{col}: {df_ratio[col].mean():.4f} ± {df_ratio[col].std():.4f}")

# 4) leave-one/two-out 패턴 감지
val_is_one  = (df_ratio["n_val"] == 1).mean()
test_is_one = (df_ratio["n_test"] == 1).mean()
print(f"\nShare of users with n_val==1:  {val_is_one:.3f}")
print(f"Share of users with n_test==1: {test_is_one:.3f}")

# 5) 60/20/20 근접도(허용 오차 5%p 예시)
close_60 = ((df_ratio["r_train"] >= 0.55) & (df_ratio["r_train"] <= 0.65)).mean()
close_20v = ((df_ratio["r_val"]   >= 0.15) & (df_ratio["r_val"]   <= 0.25)).mean()
close_20t = ((df_ratio["r_test"]  >= 0.15) & (df_ratio["r_test"]  <= 0.25)).mean()
print(f"\nUsers near 60/20/20 (±5%p): train {close_60:.3f}, val {close_20v:.3f}, test {close_20t:.3f}")

# 6) 분할 방식 히ュー리스틱 결론
conclusions = []
if val_is_one > 0.9 and test_is_one > 0.9:
    conclusions.append("패턴: leave-one-out(또는 val/test 각각 1개) 가능성 높음.")
elif close_60 > 0.8 and close_20v > 0.8 and close_20t > 0.8:
    conclusions.append("패턴: per-user 60/20/20 분할일 가능성 높음.")
else:
    conclusions.append("패턴: 균등 60/20/20 또는 leave-one 계열로 보이지 않음(글로벌 랜덤/다른 규칙 가능).")

if not (u_tr & u_va & u_te):
    conclusions.append("주의: 어떤 유저는 특정 분할에만 존재 → per-user 분할이 아닐 수도 있음.")
if dupes.any():
    conclusions.append("경고: (user,item) 중복 발견 → 데이터 누수 가능성.")

print("\n=== Heuristic Conclusion ===")
for c in conclusions: print("-", c)
