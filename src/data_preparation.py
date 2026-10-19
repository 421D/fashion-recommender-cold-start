"""
Amazon Reviews 2023 (Clothing, Shoes and Jewelry)
数据下载 + 检查 + 子采样 + 切分 + 字段质量验证脚本

完整流程，按顺序执行：
  阶段1 - stage1_download_and_inspect()
    下载官方5-core交互数据 + metadata，打印规模统计

  阶段2 - stage2_subsample()
    按时间窗口(2020年起) + 重新5-core，缩小到team电脑能跑得动的规模
    最终规模：984,067用户 / 294,021商品 / 8,509,985交互

  阶段3 - temporal_train_val_test_split() + tag_test_users_warm_cold()
    统一的时序train/val/test切分（80/10/10），并标记测试集里的
    warm用户（评估CF/Content-based/Hybrid）和cold用户（评估Knowledge-based）

  阶段4 - validate_field_quality()
    验证 CF/Content-based/Knowledge-based 各自依赖的关键字段
    （categories / bought_together / price）的完整性


依赖安装：
  pip install pandas pyarrow requests
"""

import os
import gc
import gzip
import json
import shutil
import requests
import pandas as pd

# ============ 配置 ============
CATEGORY = "Clothing_Shoes_and_Jewelry"
OUTPUT_DIR = "./processed_data"

FIVE_CORE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    f"benchmark/5core/rating_only/{CATEGORY}.csv.gz"
)
META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    f"raw/meta_categories/meta_{CATEGORY}.jsonl.gz"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# Stage 1: download official 5-core interactions + metadata, inspect scale
# =========================================================

def _download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"Already exists, skip download: {out_path}")
        return

    tmp_path = out_path + ".part"  # download to temp file first, avoid corrupted partial file on interrupt

    print(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    try:
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  progress: {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%)",
                          end="", flush=True)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if total and downloaded != total:
        os.remove(tmp_path)
        raise RuntimeError(f"Download incomplete ({downloaded}/{total} bytes), deleted, please rerun")

    os.replace(tmp_path, out_path)
    print(f"\nDownload complete: {out_path}")


META_FIELDS_TO_KEEP = [
    "parent_asin", "title", "description", "features", "categories",
    "store", "price", "bought_together", "average_rating", "rating_number",
]


def _load_meta_jsonl_filtered(gz_path, valid_items):
    """流式读取 metadata jsonl.gz，只保留匹配商品 + 只提取需要的字段（避免撑爆内存）"""
    records = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                print(f"\r  scanned {i:,} metadata records, matched {len(records):,}", end="", flush=True)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("parent_asin") in valid_items:
                slim = {k: obj.get(k) for k in META_FIELDS_TO_KEEP}
                records.append(slim)
    print(f"\r  scan complete, matched {len(records):,} records{' ' * 20}")
    return pd.DataFrame(records)


def stage1_download_and_inspect():
    gz_path = os.path.join(OUTPUT_DIR, f"{CATEGORY}_5core.csv.gz")
    csv_path = os.path.join(OUTPUT_DIR, f"{CATEGORY}_5core.csv")

    _download_file(FIVE_CORE_URL, gz_path)

    if not os.path.exists(csv_path):
        with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    reviews = pd.read_csv(csv_path)
    print(f"\nOfficial 5-core interaction data loaded: {len(reviews):,} rows")
    print("Columns:", list(reviews.columns))

    # convert user_id / parent_asin to category dtype, cuts memory usage to ~1/8
    reviews["user_id"] = reviews["user_id"].astype("category")
    reviews["parent_asin"] = reviews["parent_asin"].astype("category")

    meta_gz_path = os.path.join(OUTPUT_DIR, f"meta_{CATEGORY}.jsonl.gz")
    _download_file(META_URL, meta_gz_path)

    valid_items = set(reviews["parent_asin"].unique())
    print(f"\nParsing metadata (keeping {len(valid_items):,} target items)...")
    meta_filtered = _load_meta_jsonl_filtered(meta_gz_path, valid_items)
    print(f"Metadata items aligned with 5-core interactions: {len(meta_filtered):,}")

    # price etc. sometimes use a placeholder (e.g. '—') instead of a true null;
    # coerce to numeric, unparseable values become NaN
    for col in ["price", "average_rating", "rating_number"]:
        if col in meta_filtered.columns:
            meta_filtered[col] = pd.to_numeric(meta_filtered[col], errors="coerce")

    n_users = reviews["user_id"].nunique()
    n_items = reviews["parent_asin"].nunique()
    n_interactions = len(reviews)
    density = n_interactions / (n_users * n_items) * 100

    reviews["date"] = pd.to_datetime(reviews["timestamp"], unit="ms")

    print("\n" + "=" * 50)
    print("Stage 1 scale summary (official 5-core data, before subsampling)")
    print("=" * 50)
    print(f"Users:            {n_users:,}")
    print(f"Items:            {n_items:,}")
    print(f"Interactions:     {n_interactions:,}")
    print(f"Matrix density:   {density:.6f}%")
    print(f"Avg interactions per user: {n_interactions / n_users:.2f}")
    print(f"Avg interactions per item: {n_interactions / n_items:.2f}")
    print(f"Metadata coverage (items): {len(meta_filtered):,}")
    print(f"Time range:       {reviews['date'].min()} ~ {reviews['date'].max()}")
    print(f"Reviews memory usage: {reviews.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    if len(meta_filtered) > 0:
        print(f"Meta memory usage:    {meta_filtered.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\nInteraction count by year:")
    print(reviews["date"].dt.year.value_counts().sort_index())

    meta_filtered = meta_filtered.copy()
    meta_filtered["parent_asin"] = meta_filtered["parent_asin"].astype("category")
    if "store" in meta_filtered.columns:
        meta_filtered["store"] = meta_filtered["store"].astype("category")

    gc.collect()

    reviews_out = os.path.join(OUTPUT_DIR, f"{CATEGORY}_5core_reviews.parquet")
    meta_out = os.path.join(OUTPUT_DIR, f"{CATEGORY}_5core_meta.parquet")
    reviews.to_parquet(reviews_out, index=False)
    meta_filtered.to_parquet(meta_out, index=False)
    print(f"\nSaved:\n  {reviews_out}\n  {meta_out}")

    return reviews, meta_filtered


# =========================================================
# Stage 2: further subsample on top of official 5-core
# =========================================================

def iterative_k_core_filter(df, user_col="user_id", item_col="parent_asin", k=5):
    prev_len = -1
    round_num = 0
    while len(df) != prev_len:
        round_num += 1
        prev_len = len(df)
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
        print(f"  round {round_num}: {len(df):,} rows | {df[user_col].nunique():,} users | "
              f"{df[item_col].nunique():,} items")
    return df


def stage2_subsample(reviews, meta, min_year=None, higher_core_k=None):
    df = reviews.copy()

    if min_year is not None:
        before = len(df)
        df = df[df["date"].dt.year >= min_year]
        print(f"Filtered by year >= {min_year}: {before:,} -> {len(df):,} rows")

    if higher_core_k is not None:
        print(f"\nRaising core threshold to {higher_core_k}-core...")
        df = iterative_k_core_filter(df, k=higher_core_k)

    valid_items = set(df["parent_asin"].unique())
    meta_sub = meta[meta["parent_asin"].isin(valid_items)]

    n_users = df["user_id"].nunique()
    n_items = df["parent_asin"].nunique()
    print("\n" + "=" * 50)
    print("Stage 2 scale after subsampling")
    print("=" * 50)
    print(f"Users: {n_users:,} | Items: {n_items:,} | Interactions: {len(df):,}")

    reviews_out = os.path.join(OUTPUT_DIR, f"{CATEGORY}_subsampled_reviews.parquet")
    meta_out = os.path.join(OUTPUT_DIR, f"{CATEGORY}_subsampled_meta.parquet")
    df.to_parquet(reviews_out, index=False)
    meta_sub.to_parquet(meta_out, index=False)
    print(f"\nSaved:\n  {reviews_out}\n  {meta_out}")

    return df, meta_sub


# =========================================================
# Stage 3: unified Train / Val / Test split + warm/cold user tagging
#
# Kept here (data prep stage) so CF/Content-based/Knowledge-based don't
# have to wait for evaluation.py to be finished before they can train.
# =========================================================

def temporal_train_val_test_split(reviews: pd.DataFrame, val_frac=0.1, test_frac=0.1):
    """
    Split by time (not random): train = earlier interactions, test = more
    recent interactions. Avoids data leakage, matches the real-world
    "predict the future from the past" scenario.
    """
    df_sorted = reviews.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"Train: {len(train_df):,} rows ({train_df['date'].min()} ~ {train_df['date'].max()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['date'].min()} ~ {val_df['date'].max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['date'].min()} ~ {test_df['date'].max()})")

    return train_df, val_df, test_df


def tag_test_users_warm_cold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    warm_users: users seen in the training set -> evaluate CF/Content-based/Hybrid
    cold_users: brand-new users, not seen in training -> evaluate Knowledge-based
    """
    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())

    warm_users = test_users & train_users
    cold_users = test_users - train_users

    print(f"Total test users: {len(test_users):,}")
    print(f"  warm users (evaluate CF/Content-based/Hybrid): {len(warm_users):,} "
          f"({len(warm_users)/len(test_users)*100:.1f}%)")
    print(f"  cold users (evaluate Knowledge-based): {len(cold_users):,} "
          f"({len(cold_users)/len(test_users)*100:.1f}%)")

    return {"warm_users": warm_users, "cold_users": cold_users}


# =========================================================
# Stage 4: field quality validation
#
# Only validates the fields each method actually depends on, where a
# problem would undermine the overall architecture:
#   - categories: foundation for Content-based's category index / lookup
#   - bought_together: originally planned signal for outfit coordination
#   - price: Knowledge-based's budget filtering / refinement mechanism
# title/description/features/store only affect local quality, not the
# architecture, so they're left for each method owner to handle.
# =========================================================

def validate_field_quality(meta_df: pd.DataFrame):
    total = len(meta_df)
    print("\n" + "=" * 50)
    print("Stage 4: key field quality validation")
    print("=" * 50)

    # --- categories: foundation of Content-based architecture ---
    cat_null = meta_df["categories"].isna().sum()
    cat_empty = meta_df["categories"].apply(
        lambda x: isinstance(x, (list, tuple)) and len(x) == 0
    ).sum()
    cat_valid = total - cat_null - cat_empty
    print(f"\n[categories] total items: {total:,}")
    print(f"  null: {cat_null:,} ({cat_null/total*100:.1f}%)")
    print(f"  empty list: {cat_empty:,} ({cat_empty/total*100:.1f}%)")
    print(f"  valid category info: {cat_valid:,} ({cat_valid/total*100:.1f}%)")

    # --- bought_together: originally planned data-driven signal ---
    bt_null = meta_df["bought_together"].isna().sum()
    print(f"\n[bought_together] total items: {total:,}")
    print(f"  null: {bt_null:,} ({bt_null/total*100:.1f}%)")
    non_null_sample = meta_df[meta_df["bought_together"].notna()]["bought_together"].head(5)
    if len(non_null_sample) > 0:
        print("  non-null samples:")
        for i, val in enumerate(non_null_sample):
            print(f"    {i}: type={type(val).__name__}, value={val}")
    else:
        print("  (no non-null samples -- this field is effectively unusable in this "
              "dataset; Content-based's complementary-category mapping needs another "
              "approach, e.g. keyword bucketing + co-purchase time-window analysis)")

    # --- price: Knowledge-based budget filtering / refinement ---
    price_null = meta_df["price"].isna().sum()
    price_valid = total - price_null
    print(f"\n[price] total items: {total:,}")
    print(f"  missing: {price_null:,} ({price_null/total*100:.1f}%)")
    print(f"  valid: {price_valid:,} ({price_valid/total*100:.1f}%)")
    if price_valid > 0:
        print(meta_df["price"].describe())


if __name__ == "__main__":
    # full pipeline: download -> subsample -> split -> field validation
    reviews, meta = stage1_download_and_inspect()

    reviews_sub, meta_sub = stage2_subsample(reviews, meta, min_year=2020, higher_core_k=5)
    print(reviews_sub["date"].min(), "~", reviews_sub["date"].max())

    train_df, val_df, test_df = temporal_train_val_test_split(reviews_sub)
    train_df.to_parquet(os.path.join(OUTPUT_DIR, f"{CATEGORY}_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(OUTPUT_DIR, f"{CATEGORY}_val.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUTPUT_DIR, f"{CATEGORY}_test.parquet"), index=False)
    print("\nSaved train/val/test splits. All methods must use these same three files.")

    user_tags = tag_test_users_warm_cold(train_df, test_df)
    with open(os.path.join(OUTPUT_DIR, f"{CATEGORY}_test_user_tags.json"), "w") as f:
        json.dump(
            {"warm_users": list(user_tags["warm_users"]), "cold_users": list(user_tags["cold_users"])},
            f,
        )
    print("Saved warm/cold user tags.")

    validate_field_quality(meta_sub)