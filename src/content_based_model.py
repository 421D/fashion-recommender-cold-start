"""
content_based_model.py

Given a user's purchase history, recommends complementary
items (e.g. jacket -> pants/shoes) rather than near-duplicates. Since
bought_together is empty in this dataset, complementary categories come
from two independent strategies that can be used together:
    - Plan A (rule-based, cold-start safe): keyword bucketing + a manually
      curated complementary-bucket matrix based on fashion domain knowledge.
      Requires no interaction data, so it always has a fallback answer.
    - Plan B (data-driven): mines complementary categories from real
      co-purchase behavior (same user buying two categories within a short
      time window). More accurate where enough data exists, but sparse or
      unavailable for rare category pairs.
Plan B is preferred at recommendation time when it covers a given
category; Plan A is the fallback otherwise.
"""

from collections import defaultdict

import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _join_text_field(value) -> str:
    """
    Safely convert a metadata field (which may be a list, None, NaN, or a
    plain string) into a single whitespace-joined string.

    title/description/features/categories/store can each independently be
    a list[str], None, NaN, or a scalar string depending on the row, so a
    naive " ".join(value) call would crash on many rows. This function
    normalizes all of those cases into one consistent string form.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        # value is a list/ndarray: pd.isna() on a container raises here,
        # which just means "not a scalar NaN", so can continue normally.
        pass

    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value if v is not None and str(v) != "nan")
    return str(value)


def _get_leaf_category(categories) -> str:
    """
    Extract the most fine-grained category (the last element of the
    categories path list) for one item. Centralized here so every part of
    the module agrees on the same definition of "leaf category".
    """
    if categories is None:
        return None
    try:
        if pd.isna(categories):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(categories, (list, tuple, np.ndarray)) and len(categories) > 0:
        leaf = categories[-1]
        return str(leaf) if leaf is not None else None
    return None


# PLAN A - keyword bucketing + a bucket-level complementary-category matrix.
# LEAF_CATEGORY_KEYWORDS maps specific leaf categories (e.g. "Women's Denim
# Jackets") down to broad buckets via substring matching, so new leaf
# categories still get bucketed as long as their name contains a known
# keyword. COMPLEMENTARY_CATEGORIES then defines which buckets pair with
# which (fashion domain knowledge, e.g. jackets pair with pants/shoes).
# bucket name -> substrings to look for (case-insensitive) in a leaf
# category name. First bucket with a matching keyword wins.
LEAF_CATEGORY_KEYWORDS = {
    "Outerwear": [
        "jacket", "coat", "blazer", "parka", "vest", "windbreaker", "snow",
        "fleece",
    ],
    "Tops": [
        "shirt", "t-shirt", "tee", "blouse", "sweater", "hoodie",
        "sweatshirt", "cardigan", "top", "polo", "tank", "pullover", "tunic",
        "bodysuit", "henley", "jersey",
    ],
    "Bottoms": [
        "pant", "jean", "trouser", "short", "legging", "skirt", "chino",
        # "bottom" catches the literal leaf category named "Bottoms" itself,
        # which the original keyword list missed (a self-reference bug
        # found by inspecting the real data's Other-bucket report).
        "bottom", "tight", "overall",
    ],
    "Dresses": [
        "dress", "jumpsuit", "romper", "gown",
    ],
    "Footwear": [
        "shoe", "sneaker", "boot", "sandal", "heel", "flat", "slipper",
        "loafer", "pump", "wedge", "platform", "mule", "clog",
        "flip-flop", "flip flop", "oxford", "espadrille", "walking",
        "running", "slide",
    ],
    "Jewelry": [
        "necklace", "earring", "ring", "bracelet", "jewelry", "jewellery",
        "pendant", "anklet", "brooch", "stud", "hoop", "choker", "dangle",
        "charm", "bangle", "cuff", "chain", "bead", "statement", "link",
        "strand",
    ],
    "Bags": [
        "bag", "handbag", "backpack", "purse", "wallet", "tote", "clutch",
        "daypack", "duffel", "duffle", "satchel", "waist pack", "card case",
    ],
    "Accessories": [
        "belt", "hat", "cap", "scarf", "scarve", "glove", "sunglasses",
        "watch", "tie", "headband", "headwear", "accessor", "beanie",
        "skullie", "wig", "balaclava", "sock", "mask", "eyeglass",
        "pashmina", "fedora", "gaiter",
    ],
    "Underwear & Sleepwear": [
        "underwear", "bra", "lingerie", "pajama", "sleepwear", "boxer",
        "brief", "nightgown", "robe", "sleep",
    ],
    "Swimwear": [
        "swim", "bikini", "swimsuit", "cover-up", "cover up", "one-piece",
        "one piece", "trunk",
    ],
    "Sportswear": [
        "sport", "athletic", "activewear", "yoga", "gym", "tracksuit",
        "leg warmer",
    ],
    # A number of high-volume "Other" leaf categories were deliberately
    # left unmapped: pure descriptors like "Casual", "Formal", "Denim",
    # "Women" carry no garment-type information; "Westlake", "Chelsea" are
    # brand/style names with no general keyword; "Combat", "Suits",
    # "Mid-Calf" are genuinely ambiguous (e.g. Combat could mean boots or
    # trousers) and a wrong guess is worse than the same-category fallback.
    # "rain" was tried and rejected -- it's a substring of "training"
    # ("t-RAIN-ing"), which would silently misclassify anything with that
    # word. Any new keyword should be checked against unrelated words
    # before being added, not just against the category it's meant to fix.
}


def map_leaf_to_bucket(leaf_category) -> str:
    """
    Map one leaf category string to a broad bucket defined above. Leaf
    categories that match no keyword fall into "Other" -- these get no
    Plan A complementary suggestion and instead fall back to
    "similar items in the same category" (see _get_complementary_leaves).
    """
    if not leaf_category:
        return "Other"
    leaf_lower = str(leaf_category).lower()
    for bucket, keywords in LEAF_CATEGORY_KEYWORDS.items():
        if any(kw in leaf_lower for kw in keywords):
            return bucket
    return "Other"


# bucket -> list of buckets that make a good outfit pairing with it.
# And this is a manual, domain-knowledge matrix (not learned from data).
COMPLEMENTARY_CATEGORIES = {
    "Outerwear": ["Bottoms", "Tops", "Footwear", "Accessories"],
    "Tops": ["Bottoms", "Outerwear", "Footwear", "Accessories"],
    "Bottoms": ["Tops", "Outerwear", "Footwear", "Accessories"],
    "Dresses": ["Footwear", "Jewelry", "Bags", "Accessories", "Outerwear"],
    "Footwear": ["Tops", "Bottoms", "Outerwear", "Dresses"],
    "Jewelry": ["Dresses", "Tops", "Bottoms"],
    "Bags": ["Dresses", "Tops", "Outerwear", "Bottoms"],
    "Accessories": ["Tops", "Bottoms", "Outerwear", "Dresses"],
    "Underwear & Sleepwear": ["Underwear & Sleepwear"],
    "Swimwear": ["Accessories", "Footwear", "Bags"],
    "Sportswear": ["Footwear", "Accessories", "Bags"],
}


# PLAN B - mines category pairs from co-purchase timing in train_df, since
# bought_together is unusable (100% empty). Finer-grained than Plan A
# (leaf category, not bucket). Meant to run once offline; the output is a
# plain dict that can be cached and reused.

def build_cooccurrence_complementary_map(
    train_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    time_window_days: float = 7,
    top_k: int = 5,
    min_pair_count: int = 5,
) -> dict:
    """
    Mine a leaf-category -> [complementary leaf categories] map from
    co-purchase timing in train_df.
    """
    meta_df = meta_df.reset_index(drop=True)
    asin_to_leaf = {
        str(asin): _get_leaf_category(cats)
        for asin, cats in zip(meta_df["parent_asin"], meta_df["categories"])
    }

    df = train_df[["user_id", "parent_asin", "timestamp"]].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["leaf_category"] = df["parent_asin"].map(asin_to_leaf)
    df = df.dropna(subset=["leaf_category"])

    window_ms = time_window_days * 24 * 60 * 60 * 1000

    pair_counts = defaultdict(int)
    for _, group in df.groupby("user_id"):
        records = group.sort_values("timestamp")[["timestamp", "leaf_category"]].to_numpy()
        n = len(records)
        for i in range(n):
            t_i, cat_i = records[i]
            for j in range(i + 1, n):
                t_j, cat_j = records[j]
                if t_j - t_i > window_ms:
                    break  # records are time-sorted, later j only gets further away
                if cat_i != cat_j:
                    pair = tuple(sorted((cat_i, cat_j)))
                    pair_counts[pair] += 1

    neighbor_counts = defaultdict(lambda: defaultdict(int))
    for (cat_a, cat_b), count in pair_counts.items():
        neighbor_counts[cat_a][cat_b] += count
        neighbor_counts[cat_b][cat_a] += count

    complementary_map = {}
    for category, neighbors in neighbor_counts.items():
        strong_neighbors = {c: n for c, n in neighbors.items() if n >= min_pair_count}
        if not strong_neighbors:
            continue
        ranked = sorted(strong_neighbors.items(), key=lambda kv: kv[1], reverse=True)
        complementary_map[category] = [cat for cat, _ in ranked[:top_k]]

    return complementary_map


def _build_bucket_index(category_index: dict) -> dict:
    """
    Build {bucket_name: [leaf_category, ...]} from the leaf-level
    category_index, using Plan A's keyword mapping. Needed because
    COMPLEMENTARY_CATEGORIES is defined at bucket granularity while
    category_index is keyed by leaf category.
    """
    bucket_to_leaves = defaultdict(list)
    for leaf_category in category_index.keys():
        bucket = map_leaf_to_bucket(leaf_category)
        bucket_to_leaves[bucket].append(leaf_category)
    return dict(bucket_to_leaves)


def _get_complementary_leaves(
    query_leaf: str,
    bucket_index: dict,
    complementary_leaf_map: dict = None,
) -> list:
    """
    Resolve which leaf categories should be searched for complementary
    items, preferring Plan B (data-driven) when it has coverage for this
    leaf category, and falling back to Plan A (keyword bucket) otherwise.
    """
    if complementary_leaf_map and query_leaf in complementary_leaf_map:
        return complementary_leaf_map[query_leaf]

    query_bucket = map_leaf_to_bucket(query_leaf)
    complementary_buckets = COMPLEMENTARY_CATEGORIES.get(query_bucket, [])
    leaves = []
    for bucket in complementary_buckets:
        leaves.extend(bucket_index.get(bucket, []))
    return leaves


def _get_candidate_rows_for_item(
    query_leaf: str,
    category_index: dict,
    bucket_index: dict,
    exclude_row: int = None,
    complementary_leaf_map: dict = None,
) -> list:
    """
    Return the row indices to compare the query item against: items in
    complementary categories (Plan B if available, else Plan A), or, if
    no complementary mapping exists at all (leaf fell into "Other" and
    has no co-purchase evidence), items in the same leaf category as a
    same-category-similarity fallback.
    """
    complementary_leaves = _get_complementary_leaves(
        query_leaf, bucket_index, complementary_leaf_map
    )

    candidate_rows = []
    for leaf in complementary_leaves:
        candidate_rows.extend(category_index.get(leaf, []))

    if not candidate_rows:
        candidate_rows = list(category_index.get(query_leaf, []))

    if exclude_row is not None and exclude_row in candidate_rows:
        candidate_rows = [r for r in candidate_rows if r != exclude_row]

    return candidate_rows


def build_item_profiles(meta_df: pd.DataFrame):
    """
    Concatenate title + description + features + categories + store into
    one text blob per item and vectorize with TF-IDF.
    reset_index guarantees a contiguous 0-based row order, which is
    exactly the row-numbering scheme category_index relies on.
    """
    meta_df = meta_df.reset_index(drop=True)

    texts = []
    for _, row in meta_df.iterrows():
        title_text = _join_text_field(row.get("title"))
        desc_text = _join_text_field(row.get("description"))
        feat_text = _join_text_field(row.get("features"))
        cat_text = _join_text_field(row.get("categories"))
        store_text = _join_text_field(row.get("store"))
        texts.append(f"{title_text} {desc_text} {feat_text} {cat_text} {store_text}")

    # stop_words: drop uninformative English function words.
    # max_df/min_df: drop terms that are either near-universal (no
    # discriminative power) or appear in only one item (likely noise).
    # ngram_range=(1, 2): unigrams + bigrams, so phrases like
    # "running shoes" are captured as a unit, not just two loose words.
    # max_features: caps vocabulary size to bound memory at this scale.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Cast to str: parent_asin is a pandas "category" dtype; plain strings
    # avoid downstream dtype-compatibility surprises (dict keys, etc.).
    item_id_list = meta_df["parent_asin"].astype(str).tolist()

    return tfidf_matrix, item_id_list, vectorizer


def build_category_index(meta_df: pd.DataFrame) -> dict:
    """
    Group items by their leaf category (categories[-1]).
    Returns {leaf_category: [row_idx, ...]}, using the same row numbering
    as build_item_profiles()'s tfidf_matrix.

    This index is what makes candidate retrieval sub-linear: a query only
    ever needs to look at a few thousand rows from one or a few
    categories, never the full 294K-item catalog.
    """
    meta_df = meta_df.reset_index(drop=True)

    category_index = defaultdict(list)
    for row_idx, cats in enumerate(meta_df["categories"]):
        leaf_category = _get_leaf_category(cats)
        if leaf_category:
            category_index[leaf_category].append(row_idx)
    return dict(category_index)


def recommend_content_based(
    query_item_id: str,
    tfidf_matrix,
    item_id_list: list,
    category_index: dict,
    meta_df: pd.DataFrame,
    top_n: int = 10,
    complementary_leaf_map: dict = None,
) -> list:
    """
    Recommend items that complement a single anchor item (e.g. a jacket
    the user already owns -> recommend pants/shoes to pair with it).

    complementary_leaf_map: is the optional Plan B mapping from
    build_cooccurrence_complementary_map().
    """
    meta_df = meta_df.reset_index(drop=True)

    # item_id_list.index() is O(n); fine for a single ad-hoc query, but
    # batch callers should use get_content_based_predictions_for_all_users
    # instead, which builds an O(1) lookup table once for all users.
    try:
        query_row = item_id_list.index(str(query_item_id))
    except ValueError:
        return []  # anchor item not present in this dataset

    query_leaf = _get_leaf_category(meta_df.loc[query_row, "categories"])

    bucket_index = _build_bucket_index(category_index)
    candidate_rows = _get_candidate_rows_for_item(
        query_leaf, category_index, bucket_index,
        exclude_row=query_row, complementary_leaf_map=complementary_leaf_map,
    )
    if not candidate_rows:
        return []

    query_vector = tfidf_matrix[query_row]
    candidate_matrix = tfidf_matrix[candidate_rows]
    sims = cosine_similarity(query_vector, candidate_matrix).flatten()

    order = np.argsort(sims)[::-1][:top_n]
    recommended_rows = [candidate_rows[i] for i in order]

    return [item_id_list[r] for r in recommended_rows]


def get_content_based_predictions_for_all_users(
    tfidf_matrix,
    item_id_list: list,
    category_index: dict,
    meta_df: pd.DataFrame,
    user_histories: dict,
    top_n: int = 10,
    candidate_pools: dict = None,
    complementary_leaf_map: dict = None,
) -> dict:
    """
    Batch recommendation for many users at once.
    """
    meta_df = meta_df.reset_index(drop=True)

    # Precompute an O(1) item_id -> row lookup, shared by every user in
    # this batch (this is the main reason this function is much faster
    # than calling recommend_content_based() once per user).
    item_id_to_row = {item_id: row for row, item_id in enumerate(item_id_list)}

    if candidate_pools is None:
        leaf_by_row = [
            _get_leaf_category(meta_df.loc[row, "categories"])
            for row in range(len(meta_df))
        ]
        bucket_index = _build_bucket_index(category_index)
    else:
        leaf_by_row = None
        bucket_index = None

    predictions = {}

    for user_id, history_item_ids in user_histories.items():
        if not history_item_ids:
            # Users with zero history are true cold-start users and are
            # knowledge_based_model.py's responsibility; return an empty
            # list here rather than fabricating a recommendation.
            predictions[user_id] = []
            continue

        history_rows = set()
        valid_history_rows = []
        for item_id in history_item_ids:
            row = item_id_to_row.get(str(item_id))
            if row is not None:
                history_rows.add(row)
                valid_history_rows.append(row)

        if not valid_history_rows:
            predictions[user_id] = []
            continue

        if candidate_pools is not None:
            pool_item_ids = candidate_pools.get(user_id, [])
            candidate_rows = [
                item_id_to_row[iid] for iid in pool_item_ids
                if iid in item_id_to_row and item_id_to_row[iid] not in history_rows
            ]
        else:
            candidate_row_set = set()
            for row in valid_history_rows:
                query_leaf = leaf_by_row[row]
                candidate_row_set.update(
                    _get_candidate_rows_for_item(
                        query_leaf, category_index, bucket_index,
                        exclude_row=row, complementary_leaf_map=complementary_leaf_map,
                    )
                )
            candidate_rows = list(candidate_row_set)

        if not candidate_rows:
            predictions[user_id] = []
            continue

        # Score each candidate by the best (max) similarity across all of
        # the user's history items, rather than averaging or only using
        # the first history item.
        candidate_scores = {row: -1.0 for row in candidate_rows}
        candidate_matrix = tfidf_matrix[candidate_rows]
        for query_row in valid_history_rows:
            query_vector = tfidf_matrix[query_row]
            sims = cosine_similarity(query_vector, candidate_matrix).flatten()
            for row, sim in zip(candidate_rows, sims):
                if sim > candidate_scores[row]:
                    candidate_scores[row] = float(sim)

        sorted_rows = sorted(
            candidate_scores.keys(), key=lambda r: candidate_scores[r], reverse=True
        )

        if candidate_pools is not None:
            predictions[user_id] = [item_id_list[r] for r in sorted_rows]
        else:
            predictions[user_id] = [item_id_list[r] for r in sorted_rows[:top_n]]

    return predictions


def _build_user_histories_from_train(train_df: pd.DataFrame, user_ids=None) -> dict:
    """
    Build {user_id: [item_id, ...]} from the training interactions, i.e.
    each user's purchase history to use as anchors for content-based
    recommendation. If user_ids is given, only those users are kept
    (useful for testing on a manageable subset instead of all ~984K
    users at once).
    """
    df = train_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    if user_ids is not None:
        df = df[df["user_id"].isin(set(user_ids))]
    return df.groupby("user_id")["parent_asin"].apply(list).to_dict()


def _load_real_eval_files(candidate_pools_path: str, ground_truth_path: str,
                           user_limit: int = None):
    """
    Load the team's real candidate_pools.json / ground_truth.json

    These two files are read-only "answer sheets" shared across all four
    recommendation methods; this function never modifies them, it only
    reads and aligns them (keeps only users present in both files).
    """
    with open(candidate_pools_path) as f:
        candidate_pools = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    pool_ids = set(candidate_pools.keys())
    gt_ids = set(ground_truth.keys())
    common_users = sorted(pool_ids & gt_ids)
    print(f"candidate_pools.json users: {len(pool_ids):,}, "
          f"ground_truth.json users: {len(gt_ids):,}, overlap: {len(common_users):,}")
    if len(common_users) < min(len(pool_ids), len(gt_ids)):
        print("the two files don't fully overlap -- some users only exist in one of them, "
              "worth checking whether they were built from the same user set")

    if user_limit is not None and len(common_users) > user_limit:
        common_users = common_users[:user_limit]

    candidate_pools = {u: candidate_pools[u] for u in common_users}
    ground_truth = {u: ground_truth[u] for u in common_users}
    return candidate_pools, ground_truth


def _quick_sanity_check(predictions: dict, ground_truth: dict, k: int = 10) -> dict:
    """
    NOT the official evaluation metric. The team's real precision_at_k /
    recall_at_k / ndcg_at_k / evaluate_model() live in evaluation.py and
    are the source of truth for the cross-method comparison table -- this
    is only a fast, local self-check (Hit Rate@k and MRR) so you can see
    at a glance whether the ranking looks reasonable before that's ready.
    """
    hits = 0
    reciprocal_ranks = []
    evaluated = 0

    for user_id, true_items in ground_truth.items():
        ranked = predictions.get(user_id)
        if not ranked:
            continue
        evaluated += 1
        true_set = set(true_items)

        if any(item in true_set for item in ranked[:k]):
            hits += 1

        rr = 0.0
        for rank, item in enumerate(ranked, start=1):
            if item in true_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "evaluated_users": evaluated,
        "hit_rate_at_k": hits / evaluated if evaluated else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "k": k,
    }


def _find_data_file(filename: str) -> str:
    """
    Locate a data file by checking a few likely locations, so this script
    still works whether it's run from the project root, from inside
    processed_data/, or from an IDE that sets a different working
    directory. Raises FileNotFoundError with a clear message listing all
    paths tried if none of them exist.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), "processed_data"),
        script_dir,
        os.path.join(script_dir, "processed_data"),
    ]
    tried = []
    for d in candidate_dirs:
        candidate = os.path.join(d, filename)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find '{filename}'. Tried:\n" + "\n".join(f"  - {p}" for p in tried) +
        "\nMake sure processed_data/ is next to this script, or run this "
        "script from the folder that contains processed_data/."
    )


if __name__ == "__main__":
    COOC_USER_SAMPLE = 50000   # users used to mine Plan B; None = use all of train_df
    REAL_EVAL_USER_LIMIT = None  # users evaluated against real files; None = all of them
    OUTPUT_DIR = "./content_based_outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    meta = pd.read_parquet(
        _find_data_file("Clothing_Shoes_and_Jewelry_subsampled_meta.parquet")
    )

    # how much of the catalog falls into "Other" (no Plan A keyword match)
    leaf_counts = meta["categories"].apply(
        lambda x: x[-1] if x is not None and len(x) > 0 else None
    ).value_counts()
    bucket_counts = leaf_counts.rename(index=map_leaf_to_bucket).groupby(level=0).sum()
    print("Item count per bucket (Plan A coverage check):")
    print(bucket_counts.sort_values(ascending=False))

    print("\nBuilding TF-IDF item profiles ...")
    tfidf_matrix, item_ids, vectorizer = build_item_profiles(meta)
    print(f"tfidf_matrix shape: {tfidf_matrix.shape}")

    print("Building category index ...")
    category_index = build_category_index(meta)
    print(f"{len(category_index)} leaf categories")

    try:
        train_df = pd.read_parquet(
            _find_data_file("Clothing_Shoes_and_Jewelry_train.parquet")
        )
        if COOC_USER_SAMPLE is not None:
            rng = random.Random(42)
            all_train_users = train_df["user_id"].astype(str).unique().tolist()
            sampled_users = set(
                rng.sample(all_train_users, min(COOC_USER_SAMPLE, len(all_train_users)))
            )
            cooc_train_df = train_df[train_df["user_id"].astype(str).isin(sampled_users)]
            print(f"\n[Plan B] mining from {len(sampled_users):,} sampled users "
                  f"({len(cooc_train_df):,} interactions) ...")
        else:
            cooc_train_df = train_df
            print(f"\n[Plan B] mining from all {train_df['user_id'].nunique():,} users ...")

        _t0 = time.time()
        cooc_map = build_cooccurrence_complementary_map(
            cooc_train_df, meta, time_window_days=7, top_k=5, min_pair_count=5
        )
        print(f"[Plan B] covered {len(cooc_map)} / {len(category_index)} leaf categories "
              f"({time.time() - _t0:.1f}s). Categories with no Plan B coverage fall back "
              f"to Plan A.")
    except FileNotFoundError:
        train_df = None
        cooc_map = None
        print("\n[Plan B] train parquet not found, Plan A only.")

    # quick smoke test on one item, to confirm Plan A and Plan B are wired up correctly
    sample_id = item_ids[0]
    recs_plan_a = recommend_content_based(
        sample_id, tfidf_matrix, item_ids, category_index, meta, top_n=5
    )
    print(f"\n[Plan A only] recommendations for anchor {sample_id}:", recs_plan_a)
    if cooc_map:
        recs_plan_ab = recommend_content_based(
            sample_id, tfidf_matrix, item_ids, category_index, meta, top_n=5,
            complementary_leaf_map=cooc_map,
        )
        print(f"[Plan A+B] recommendations for anchor {sample_id}:", recs_plan_ab)

    # evaluation using candidate_pools.json and ground_truth.json
    try:
        pools_path = _find_data_file("candidate_pools.json")
        gt_path = _find_data_file("ground_truth.json")
    except FileNotFoundError:
        pools_path = None
        gt_path = None

    if pools_path and gt_path and train_df is not None:
        print(f"\n[real eval] found candidate_pools.json and ground_truth.json, using them ...")
        candidate_pools, ground_truth = _load_real_eval_files(
            pools_path, gt_path, user_limit=REAL_EVAL_USER_LIMIT
        )
        print(f"evaluating on {len(candidate_pools):,} users "
              f"(REAL_EVAL_USER_LIMIT={REAL_EVAL_USER_LIMIT})")

        eval_user_histories = _build_user_histories_from_train(
            train_df, user_ids=list(candidate_pools.keys())
        )
        n_no_history = len(candidate_pools) - len(eval_user_histories)
        if n_no_history:
            print(f"{n_no_history:,} of these users have no interactions in train_df "
                  f"and get dropped here (they won't appear in the output at all)")

        _t0 = time.time()
        eval_predictions = get_content_based_predictions_for_all_users(
            tfidf_matrix, item_ids, category_index, meta, eval_user_histories,
            candidate_pools=candidate_pools,
        )
        print(f"ranked candidate pools for all users in {time.time() - _t0:.1f}s")

        # save output
        out_path = os.path.join(OUTPUT_DIR, "content_based_predictions.json")
        with open(out_path, "w") as f:
            json.dump(eval_predictions, f)
        print(f"predictions saved to {out_path} "
              f"(this is the `predictions` argument evaluate_model() expects)")

        sanity = _quick_sanity_check(eval_predictions, ground_truth, k=10)
        print(f"\n[self-check only, not evaluation.py's metric] "
              f"Hit Rate@{sanity['k']} = {sanity['hit_rate_at_k']:.4f}, "
              f"MRR = {sanity['mrr']:.4f} (over {sanity['evaluated_users']:,} users)")
    else:
        print("\n[real eval] candidate_pools.json / ground_truth.json not found "
              "next to the other data files, skipping real evaluation.")

    # confirms candidate_pools mode returns the full ranked pool, not just top_n
    fake_histories = {"user_a": [item_ids[0]], "user_b": [item_ids[1], item_ids[2]]}
    fake_pool = {"user_a": [item_ids[10], item_ids[20], item_ids[30], item_ids[0]]}
    pool_recs = get_content_based_predictions_for_all_users(
        tfidf_matrix, item_ids, category_index, meta, fake_histories,
        candidate_pools=fake_pool,
    )
    print("\n[evaluation-mode smoke test] full ranked candidate pool:", pool_recs)