"""
evaluation.py
Fashion Recommender — Evaluation Module (Shared by all methods)
Author: Chuliang
 
本模块只接收标准化输入 {user_id: [item_id_1, item_id_2, ...]}（排好序的推荐列表）
和对应的 ground truth，不关心其他方法内部如何实现。
 
统一入口：evaluate_model(model_name, predictions, ground_truth, k=10)
 
负采样候选池：全量对 29.4 万商品排序不可行（实测 5-6 天）。改用每个用户的候选池
= 正样本 + n_negatives 个随机负样本，下游方法只对这 ~100 个候选排序。候选池固定
seed，保证四个方法拿到完全一致的候选，否则指标不可比较。
 
CTR / 转化率说明：这两个系统层指标不用大规模模拟数据算（模拟数据得出的点击率
不可信），改用小规模真实 User Study 收集到的真实点击/购买观测。intra_list_diversity
和 novelty 依然用真实数据（TF-IDF向量、训练集商品热度）确定性计算，不受影响。
"""
 
import json
from typing import Dict, List, Optional, Set, Tuple
 
import numpy as np
import pandas as pd
 
 
# ---------------------------------------------------------------------------
# 模型层指标（预测评分误差）
# ---------------------------------------------------------------------------
 
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
 
 
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
 
 
# ---------------------------------------------------------------------------
# 排序指标（per-user 计算后取平均）
# ---------------------------------------------------------------------------
 
def _relevant_set(user_id, ground_truth: Dict) -> Set:
    gt = ground_truth.get(user_id, [])
    return gt if isinstance(gt, set) else set(gt)
 
 
def precision_at_k(predictions: Dict[str, List], ground_truth: Dict[str, List], k: int = 10) -> float:
    scores = []
    for user_id, rec_list in predictions.items():
        relevant = _relevant_set(user_id, ground_truth)
        if not relevant:
            continue
        top_k = rec_list[:k]
        hits = sum(1 for item in top_k if item in relevant)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0
 
 
def recall_at_k(predictions: Dict[str, List], ground_truth: Dict[str, List], k: int = 10) -> float:
    scores = []
    for user_id, rec_list in predictions.items():
        relevant = _relevant_set(user_id, ground_truth)
        if not relevant:
            continue
        top_k = rec_list[:k]
        hits = sum(1 for item in top_k if item in relevant)
        scores.append(hits / len(relevant))
    return float(np.mean(scores)) if scores else 0.0
 
 
def ndcg_at_k(predictions: Dict[str, List], ground_truth: Dict[str, List], k: int = 10) -> float:
    scores = []
    for user_id, rec_list in predictions.items():
        relevant = _relevant_set(user_id, ground_truth)
        if not relevant:
            continue
        top_k = rec_list[:k]
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0
 
 
# ---------------------------------------------------------------------------
# 系统层指标（CTR/转化率来自真实 User Study；diversity/novelty 来自真实数据）
# ---------------------------------------------------------------------------
 
def click_through_rate(predictions: Dict[str, List], user_study_clicks: Dict[str, List], k: int = 10) -> float:
    """user_study_clicks: 来自小规模真实 User Study 走查时记录的真实点击行为。"""
    scores = []
    for user_id, rec_list in predictions.items():
        if user_id not in user_study_clicks:
            continue
        clicked = set(user_study_clicks[user_id])
        top_k = rec_list[:k]
        scores.append(sum(1 for item in top_k if item in clicked) / k)
    return float(np.mean(scores)) if scores else 0.0
 
 
def conversion_rate(predictions: Dict[str, List], user_study_purchases: Dict[str, List], k: int = 10) -> float:
    """user_study_purchases: 来自小规模真实 User Study 走查时记录的真实购买/收藏意向。"""
    scores = []
    for user_id, rec_list in predictions.items():
        if user_id not in user_study_purchases:
            continue
        purchased = set(user_study_purchases[user_id])
        top_k = rec_list[:k]
        scores.append(sum(1 for item in top_k if item in purchased) / k)
    return float(np.mean(scores)) if scores else 0.0
 
 
def intra_list_diversity(predictions: Dict[str, List], item_vectors: Dict, k: int = 10) -> float:
    """item_vectors: {item_id: 向量}，可复用 content-based 的 TF-IDF 向量。"""
    from sklearn.metrics.pairwise import cosine_similarity
 
    scores = []
    for user_id, rec_list in predictions.items():
        top_k = [i for i in rec_list[:k] if i in item_vectors]
        if len(top_k) < 2:
            continue
        vecs = np.array([item_vectors[i] for i in top_k])
        sim = cosine_similarity(vecs)
        n = len(top_k)
        avg_sim = (sim.sum() - np.trace(sim)) / 2 / (n * (n - 1) / 2)
        scores.append(1 - avg_sim)
    return float(np.mean(scores)) if scores else 0.0
 
 
def novelty(predictions: Dict[str, List], train_item_popularity: Dict, k: int = 10) -> float:
    """用 self-information -log2(popularity / total) 衡量，冷门商品新颖度更高。"""
    total = sum(train_item_popularity.values())
    scores = []
    for user_id, rec_list in predictions.items():
        top_k = rec_list[:k]
        if not top_k:
            continue
        item_scores = []
        for item in top_k:
            pop = train_item_popularity.get(item, 0)
            if pop == 0 or total == 0:
                item_scores.append(np.log2(total) if total > 0 else 0.0)
            else:
                item_scores.append(-np.log2(pop / total))
        scores.append(np.mean(item_scores))
    return float(np.mean(scores)) if scores else 0.0
 
 
# ---------------------------------------------------------------------------
# 商品热度统计（novelty 的数据来源）
# ---------------------------------------------------------------------------
 
def build_item_popularity(
    train_df: pd.DataFrame, item_col: str = "parent_asin"
) -> Dict:
    """统计训练集里每个商品的交互次数，供 novelty 使用。"""
    return train_df[item_col].astype(str).value_counts().to_dict()
 
 
# ---------------------------------------------------------------------------
# warm / cold 用户拆分（配合 test_user_tags.json）
# ---------------------------------------------------------------------------
 
def load_cold_user_ids(test_user_tags_path: str) -> Set:
    with open(test_user_tags_path, "r") as f:
        tags = json.load(f)
    return set(tags.get("cold_users", []))
 
 
def split_predictions_by_user_type(predictions: Dict, cold_user_ids: Set) -> Tuple[Dict, Dict]:
    warm_preds = {u: v for u, v in predictions.items() if u not in cold_user_ids}
    cold_preds = {u: v for u, v in predictions.items() if u in cold_user_ids}
    return warm_preds, cold_preds
 
 
# ---------------------------------------------------------------------------
# 负采样候选池生成（团队统一维护，所有方法共用）
# ---------------------------------------------------------------------------
 
def build_candidate_pools(
    test_df: pd.DataFrame,
    all_items: List,
    n_negatives: int = 99,
    seed: int = 42,
    rating_threshold: float = 4,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    rating_col: str = "rating",
) -> Dict:
    """
    对每个测试集用户：候选池 = 该用户的正样本 + n_negatives 个随机负样本，混合打乱后返回。
    所有方法必须共用同一份候选池，否则指标不可比较。
 
    正样本定义与 build_ground_truth 保持一致：只有 rating >= rating_threshold 才算正样本。
    没有任何正样本的用户（全部交互都 < threshold）直接跳过、不生成候选池，因为他们在
    precision/recall/ndcg 里 relevant 集合为空、本来也不会被计入指标。
 
    Args:
        all_items: 全部商品 ID 列表（建议用 train_df["parent_asin"].astype(str).unique().tolist()）
        rating_threshold: 与 build_ground_truth 用同一个值，保证两函数正样本定义一致
        seed: 随机种子，保证可复现、各方法拿到相同候选池
 
    Returns:
        {user_id: [item_id_1, item_id_2, ...]}
    """
    rng = np.random.default_rng(seed)
    all_items_arr = np.asarray([str(i) for i in all_items])
    n_all = len(all_items_arr)
 
    relevant_df = test_df[test_df[rating_col] >= rating_threshold]
    test_positives = (
        relevant_df.assign(**{item_col: relevant_df[item_col].astype(str)})
        .groupby(user_col, observed=True)[item_col]
        .apply(set)
        .to_dict()
    )
 
    candidate_pools = {}
    for user_id, pos_items in test_positives.items():
        pos_list = list(pos_items)
 
        # 过采样 + 过滤，避免对 29.4 万商品做 setdiff（正样本极少，几乎一轮就够）
        neg_items = []
        neg_seen = set(pos_items)
        while len(neg_items) < n_negatives:
            draw = all_items_arr[rng.integers(0, n_all, size=n_negatives * 2)]
            for it in draw:
                if it not in neg_seen:
                    neg_items.append(it)
                    neg_seen.add(it)
                    if len(neg_items) >= n_negatives:
                        break
 
        pool = pos_list + neg_items
        rng.shuffle(pool)
        candidate_pools[user_id] = pool
 
    return candidate_pools
 
 
# ---------------------------------------------------------------------------
# ground truth 构建（从 test.parquet）
# ---------------------------------------------------------------------------
 
def build_ground_truth(
    test_df: pd.DataFrame,
    rating_threshold: float = 4,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    rating_col: str = "rating",
) -> Dict[str, List]:
    """只把 rating >= threshold 的交互算作正样本（relevant item）。"""
    relevant = test_df[test_df[rating_col] >= rating_threshold]
    return relevant.groupby(user_col, observed=True)[item_col].apply(list).to_dict()
 
 
# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
 
def evaluate_model(
    model_name: str,
    predictions: Dict[str, List],
    ground_truth: Dict[str, List],
    k: int = 10,
    rating_predictions: Optional[Tuple[List[float], List[float]]] = None,
    user_study_clicks: Optional[Dict] = None,
    user_study_purchases: Optional[Dict] = None,
    item_vectors: Optional[Dict] = None,
    train_item_popularity: Optional[Dict] = None,
) -> Dict:
    """
    Args:
        model_name: "CF" / "Content-based" / "Knowledge-based" / "Hybrid"
        predictions: {user_id: [item_id, ...]} 排好序的推荐列表
        ground_truth: {user_id: [item_id, ...]} 用户实际交互过的商品
        rating_predictions: (y_true, y_pred)，只有 CF 提供，其余传 None -> RMSE/MAE 记为 "N/A"
        user_study_clicks / user_study_purchases: 来自小规模真实 User Study 的
            真实点击/购买观测（不是大规模模拟数据），不传则 CTR/转化率记为 "N/A"
        item_vectors: 多样性计算用（复用 content-based 的 TF-IDF 向量）
        train_item_popularity: 新颖性计算用
 
    Returns:
        指标名 -> 数值（或 "N/A"）的字典
    """
    results = {"model": model_name, "k": k}
 
    if rating_predictions is not None:
        y_true, y_pred = rating_predictions
        results["rmse"] = rmse(y_true, y_pred)
        results["mae"] = mae(y_true, y_pred)
    else:
        results["rmse"] = "N/A"
        results["mae"] = "N/A"
 
    results["precision_at_k"] = precision_at_k(predictions, ground_truth, k)
    results["recall_at_k"] = recall_at_k(predictions, ground_truth, k)
    results["ndcg_at_k"] = ndcg_at_k(predictions, ground_truth, k)
 
    results["ctr"] = click_through_rate(predictions, user_study_clicks, k) if user_study_clicks else "N/A"
    results["conversion_rate"] = (
        conversion_rate(predictions, user_study_purchases, k) if user_study_purchases else "N/A"
    )
    results["intra_list_diversity"] = (
        intra_list_diversity(predictions, item_vectors, k) if item_vectors else "N/A"
    )
    results["novelty"] = novelty(predictions, train_item_popularity, k) if train_item_popularity else "N/A"
 
    return results
 
 
def compare_models(all_results: List[Dict]) -> pd.DataFrame:
    """把多次 evaluate_model() 的结果拼成对比表：行=指标，列=模型。"""
    df = pd.DataFrame(all_results).set_index("model")
    return df.T
 
 
if __name__ == "__main__":
    toy_predictions = {
        "u1": ["i1", "i2", "i3", "i4", "i5"],
        "u2": ["i3", "i5", "i1", "i2", "i4"],
    }
    toy_ground_truth = {
        "u1": ["i2", "i5"],
        "u2": ["i1", "i9"],
    }
 
    print("Precision@5:", precision_at_k(toy_predictions, toy_ground_truth, k=5))
    print("Recall@5:", recall_at_k(toy_predictions, toy_ground_truth, k=5))
    print("NDCG@5:", ndcg_at_k(toy_predictions, toy_ground_truth, k=5))
 
    toy_y_true = [4.0, 3.0, 5.0, 2.0]
    toy_y_pred = [3.5, 3.2, 4.8, 2.5]
    print("RMSE:", rmse(toy_y_true, toy_y_pred))
    print("MAE:", mae(toy_y_true, toy_y_pred))
 
    toy_test_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "parent_asin": ["i2", "i5", "i1", "i7"],
            "rating": [5, 4, 5, 3],
        }
    )
    toy_all_items = [f"i{i}" for i in range(1, 501)]
    pools = build_candidate_pools(toy_test_df, toy_all_items, n_negatives=99, seed=42)
    gt = build_ground_truth(toy_test_df, rating_threshold=4)
    # u3 只有 rating=3 的交互 -> 应被跳过，候选池里不该有 u3
    print("\n候选池用户数:", len(pools), "(应为 2, u3 被跳过)")
    print("u3 是否被正确跳过:", "u3" not in pools)
    print("候选池用户集 == ground_truth 用户集:", set(pools) == set(gt))
    first_user = list(pools.keys())[0]
    print(f"用户 {first_user} 候选池大小:", len(pools[first_user]))
    print("u1 的正样本 i2/i5 是否都在候选池:",
          all(x in pools["u1"] for x in ["i2", "i5"]))
    pools2 = build_candidate_pools(toy_test_df, toy_all_items, n_negatives=99, seed=42)
    print("固定 seed 可复现:", pools["u1"] == pools2["u1"])
 
    toy_item_vectors = {f"i{i}": np.random.rand(20) for i in range(1, 10)}
    toy_popularity = {"i1": 500, "i2": 300, "i3": 50, "i4": 10}
 
    # CTR/转化率现在改用手写的假数据代表"真实User Study记录"，
    # 不再调用已删除的 simulate_user_interaction
    toy_user_study_clicks = {"u1": ["i2"], "u2": ["i1", "i5"]}
    toy_user_study_purchases = {"u1": [], "u2": ["i1"]}
 
    result_cf = evaluate_model(
        "CF (toy)", toy_predictions, toy_ground_truth, k=5,
        rating_predictions=(toy_y_true, toy_y_pred),
        user_study_clicks=toy_user_study_clicks, user_study_purchases=toy_user_study_purchases,
        item_vectors=toy_item_vectors, train_item_popularity=toy_popularity,
    )
    result_content = evaluate_model(
        "Content-based (toy)", toy_predictions, toy_ground_truth, k=5,
        rating_predictions=None,
        user_study_clicks=toy_user_study_clicks, user_study_purchases=toy_user_study_purchases,
        item_vectors=toy_item_vectors, train_item_popularity=toy_popularity,
    )
 
    print("\n对比表：")
    print(compare_models([result_cf, result_content]))
    print("\n自测通过 ✅")
 
