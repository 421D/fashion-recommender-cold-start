"""
hybrid_model.py — owner: Yuxuan
 
Method: Hybrid (Switching + Weighted via Reciprocal Rank Fusion)
 
设计思路：
  - Switching: 按用户是否在训练集里有交互历史，分成 cold / warm 两类
      cold（训练集零历史）→ 只能用 Knowledge-based
      warm（训练集有历史）→ 用 CF + Content-based + Global Popularity
      三者的 Weighted 融合结果
 
  - Weighted: 用 Reciprocal Rank Fusion (RRF)，不是直接加权分数。
      原因：CF的打分（预测评分1-5）、Content-based的打分（余弦相似度0-1）、
      Popularity的打分（交互次数，量级可能是几千）三者尺度完全不同，直接
      线性加权会被尺度差异误导。RRF只看每个方法给出的排名位置，用排名倒数
      加权融合，避开了"跨方法分数不可比"这个问题，是推荐系统融合排序的
      标准做法（Cormack et al., 2009）。
 
  - 为什么加入 Popularity 作为第三个信号：负采样评估协议下，Popularity类
      baseline的Recall@10（41-44%）明显高于CF/Content-based（25-27%），
      这是因为测试集正样本本身就更可能是热门商品（详见报告Analysis部分）。
      给Hybrid加入热度信号，是希望在保留个性化能力的同时，部分缩小这个
      差距——但热度权重不能设太高，否则Hybrid会退化成"热度为主"，丢失
      个性化推荐的意义，最终配置为 CF 0.3 / Content-based 0.3 / Popularity 0.4（见报告 5.4 节）。
 
统一接口约定（跟团队其他模块一致）：
  - 输入的 cf_predictions / content_predictions / knowledge_predictions /
    popularity_predictions 都是 {user_id: [item_id_1, item_id_2, ...]}
    （按排名从高到低排好序的候选池全部结果，不是只有top_10）
  - 输出同样是 {user_id: [item_id...]}，候选池模式下返回全部排序结果，
    不截断——跟团队其他模块保持一致的约定，截断到 k 交给 evaluate_model()
    统一处理
"""
 
from typing import Dict, List, Optional
 
 
def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
) -> List[str]:
    """
    把多个方法给出的排名列表，融合成一份排名。
 
    ranked_lists: 每个方法各自给出的、按排名从高到低排好序的商品ID列表
    weights: 每个方法的权重，默认所有方法等权重
    rrf_k: RRF公式里的平滑常数，60是文献里常用的默认值
 
    公式: fused_score(item) = sum_over_methods( weight_m / (rrf_k + rank_in_method_m) )
    某个方法没推荐到的商品，视为该方法对这个商品没有贡献（不是0分，是不参与求和）
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 的长度必须和 ranked_lists 一致")
 
    fused_scores: Dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked_list):
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + weight / (rrf_k + rank + 1)
 
    return sorted(fused_scores.keys(), key=lambda item_id: fused_scores[item_id], reverse=True)
 
 
def weighted_hybrid(
    cf_ranked: List[str],
    content_ranked: List[str],
    popularity_ranked: Optional[List[str]] = None,
    weight_cf: float = 0.3,
    weight_content: float = 0.3,
    weight_popularity: float = 0.4,
    rrf_k: int = 60,
) -> List[str]:
    """
    单个用户的 CF + Content-based (+ Popularity) 融合结果（RRF方式）。
 
    popularity_ranked 是可选的——不传的话，退化成只融合 CF + Content-based
    （跟之前没加热度信号时的行为一致）。
    """
    if popularity_ranked is None:
        return reciprocal_rank_fusion(
            [cf_ranked, content_ranked], weights=[weight_cf, weight_content], rrf_k=rrf_k
        )
    return reciprocal_rank_fusion(
        [cf_ranked, content_ranked, popularity_ranked],
        weights=[weight_cf, weight_content, weight_popularity],
        rrf_k=rrf_k,
    )
 
 
def switching_hybrid(
    user_id: str,
    cold_user_ids: set,
    cf_predictions: Dict[str, List[str]],
    content_predictions: Dict[str, List[str]],
    knowledge_predictions: Dict[str, List[str]],
    popularity_predictions: Optional[Dict[str, List[str]]] = None,
    weight_cf: float = 0.3,
    weight_content: float = 0.3,
    weight_popularity: float = 0.4,
) -> List[str]:
    """
    单个用户的switching决策：cold用户走Knowledge-based，
    warm用户走Weighted(CF + Content-based [+ Popularity])。
    """
    if user_id in cold_user_ids:
        return knowledge_predictions.get(user_id, [])
 
    cf_ranked = cf_predictions.get(user_id, [])
    content_ranked = content_predictions.get(user_id, [])
    popularity_ranked = (
        popularity_predictions.get(user_id, []) if popularity_predictions else None
    )
 
    if not cf_ranked and not content_ranked:
        return popularity_ranked or []
    if not content_ranked:
        return cf_ranked
    if not cf_ranked:
        return content_ranked
 
    return weighted_hybrid(
        cf_ranked, content_ranked, popularity_ranked,
        weight_cf, weight_content, weight_popularity,
    )
 
 
def get_hybrid_predictions(
    target_users: List[str],
    cold_user_ids: set,
    cf_predictions: Dict[str, List[str]],
    content_predictions: Dict[str, List[str]],
    knowledge_predictions: Dict[str, List[str]],
    popularity_predictions: Optional[Dict[str, List[str]]] = None,
    weight_cf: float = 0.3,
    weight_content: float = 0.3,
    weight_popularity: float = 0.4,
) -> Dict[str, List[str]]:
    """
    批量生成Hybrid推荐结果，覆盖 target_users 里的每一个用户。
    """
    predictions = {}
    for user_id in target_users:
        predictions[user_id] = switching_hybrid(
            user_id, cold_user_ids, cf_predictions, content_predictions,
            knowledge_predictions, popularity_predictions,
            weight_cf, weight_content, weight_popularity,
        )
    return predictions
 
 
if __name__ == "__main__":
    # 自测：用假数据验证switching + 三方RRF融合逻辑
    mock_cold_ids = {"u_cold_1"}
    mock_cf = {"u_warm_1": ["itemA", "itemB", "itemC", "itemD"]}
    mock_content = {"u_warm_1": ["itemB", "itemA", "itemE", "itemC"]}
    mock_popularity = {"u_warm_1": ["itemF", "itemA", "itemB", "itemG"]}
    mock_knowledge = {"u_cold_1": ["itemX", "itemY", "itemZ"]}
 
    cold_result = switching_hybrid(
        "u_cold_1", mock_cold_ids, mock_cf, mock_content, mock_knowledge, mock_popularity
    )
    print(f"cold用户走KB: {cold_result}")
 
    warm_result_no_pop = switching_hybrid(
        "u_warm_1", mock_cold_ids, mock_cf, mock_content, mock_knowledge, None
    )
    print(f"warm用户(无热度信号): {warm_result_no_pop}")
 
    warm_result_with_pop = switching_hybrid(
        "u_warm_1", mock_cold_ids, mock_cf, mock_content, mock_knowledge, mock_popularity
    )
    print(f"warm用户(加热度信号): {warm_result_with_pop}")
    # itemA/itemB三个方法都出现且排名靠前，加了热度信号后应该排名更靠前/更稳固
 
