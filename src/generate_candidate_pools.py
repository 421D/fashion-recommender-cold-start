"""
generate_candidate_pools.py
分批版本：把候选池生成拆成一批批处理，每批处理完立刻写入文件、清空内存，
不会像原版那样把24万+用户的候选池全部同时攒在内存里。
 
跟 evaluation.py 里的 build_candidate_pools 逻辑完全一致（同样的负采样规则、
同样的 seed=42），只是拆成分批写入，最终结果应该和不分批跑出来的完全一样，
不影响正确性，只是为了降低跑的时候的内存峰值。
 
用法：
    python generate_candidate_pools.py
"""
 
import gc
import json
import numpy as np
import pandas as pd
 
BATCH_SIZE = 20000  # 每批处理多少个用户，内存还是紧张可以调小，比如10000
N_NEGATIVES = 99
SEED = 42
RATING_THRESHOLD = 4
 
# ---------------------------------------------------------------------------
# 1. 读 train.parquet，只为了提取 all_items，提取完立刻释放
# ---------------------------------------------------------------------------
print("读取 train.parquet ...")
train_df = pd.read_parquet("processed_data/Clothing_Shoes_and_Jewelry_train.parquet")
all_items_arr = np.asarray(train_df["parent_asin"].astype(str).unique())
n_all = len(all_items_arr)
print(f"all_items 提取完成，共 {n_all:,} 个商品")
 
del train_df
gc.collect()
print("已释放 train_df，内存已回收")
 
# ---------------------------------------------------------------------------
# 2. 读 test.parquet，算出每个用户的正样本（rating >= threshold）
#    这个字典本身不大（只存正样本，不是完整候选池），可以放心留在内存里
# ---------------------------------------------------------------------------
print("读取 test.parquet ...")
test_df = pd.read_parquet("processed_data/Clothing_Shoes_and_Jewelry_test.parquet")
 
relevant_df = test_df[test_df["rating"] >= RATING_THRESHOLD]
test_positives = (
    relevant_df.assign(parent_asin=relevant_df["parent_asin"].astype(str))
    .groupby("user_id", observed=True)["parent_asin"]
    .apply(set)
    .to_dict()
)
print(f"有正样本的用户数: {len(test_positives):,}")
 
del test_df, relevant_df
gc.collect()
 
# ---------------------------------------------------------------------------
# 3. 分批生成候选池，每批处理完立刻写入文件、清空这批的内存
#    用同一个 rng 对象贯穿全程（不是每批重新建），保证结果和不分批跑
#    完全一致，只是内存峰值降低
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
user_ids = list(test_positives.keys())  # 固定顺序，保证可复现
n_users = len(user_ids)
n_batches = (n_users + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\n开始分批生成候选池，共 {n_users:,} 用户，分 {n_batches} 批，每批 {BATCH_SIZE:,} 人")
 
with open("candidate_pools.json", "w") as f:
    f.write("{")
    first_entry = True
 
    for batch_idx in range(n_batches):
        batch_users = user_ids[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_pools = {}
 
        for user_id in batch_users:
            pos_items = test_positives[user_id]
            pos_list = list(pos_items)
 
            neg_items = []
            neg_seen = set(pos_items)
            while len(neg_items) < N_NEGATIVES:
                draw = all_items_arr[rng.integers(0, n_all, size=N_NEGATIVES * 2)]
                for it in draw:
                    if it not in neg_seen:
                        neg_items.append(it)
                        neg_seen.add(it)
                        if len(neg_items) >= N_NEGATIVES:
                            break
 
            pool = pos_list + neg_items
            rng.shuffle(pool)
            batch_pools[user_id] = pool.tolist() if isinstance(pool, np.ndarray) else list(pool)
 
        # 这一批写进文件，然后整批从内存里清掉
        for user_id, pool in batch_pools.items():
            if not first_entry:
                f.write(",")
            f.write(json.dumps(user_id) + ":" + json.dumps(pool))
            first_entry = False
 
        del batch_pools
        gc.collect()
 
        done = min((batch_idx + 1) * BATCH_SIZE, n_users)
        print(f"  第 {batch_idx + 1}/{n_batches} 批完成，累计处理 {done:,}/{n_users:,} 用户")
 
    f.write("}")
 
print("\ncandidate_pools.json 生成完成")
 
# ---------------------------------------------------------------------------
# 4. ground truth（这个字典很小，不用分批）
# ---------------------------------------------------------------------------
ground_truth = {uid: list(pos) for uid, pos in test_positives.items()}
with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f)
 
print("ground_truth.json 生成完成")
print(f"\n用户集一致性: {set(ground_truth.keys()) == set(user_ids)}")
print("\n已保存 candidate_pools.json 和 ground_truth.json，发群里即可。")
print("四个方法读法：")
print("  import json")
print("  candidate_pools = json.load(open('candidate_pools.json'))")
print("  # 然后只对 candidate_pools[user_id] 里的商品排序")
 
