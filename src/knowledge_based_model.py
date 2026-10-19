"""
knowledge_based_model.py — owner: Danni Cai

Method: Knowledge-based Recommendation (handles cold-start users via onboarding rules)


1. user_preferences dict needs these exact keys: style / budget_max / brands / excluded_styles.
   Wrong field names won't crash the function — it'll silently treat it as "no preference given"
   and return results that look fine but are wrong. Double check field names line up.
2. Single-user real-time calls: use recommend_knowledge_based() directly.
   Batch calls: use get_knowledge_based_predictions_for_new_users().
3. excluded_styles only suppresses the style score in single-user calls. Hard exclusion of
   items matching excluded style keywords only happens in the batch function.
4. Supports candidate_pool (single-user) / candidate_pools (batch) to restrict recommendations
   to each user's evaluation candidate set. Full batch on real data: ~5.2 min for 27,573 users.
5. Known limitation: fallback logic still searches the full catalog, not the candidate pool.
   Tested with 0% fallback trigger rate on real data, so it hasn't caused issues so far —
   worth re-checking if candidate pools get smaller or constraints get stricter.
"""

import numpy as np
import pandas as pd
import random
import json
import time
import warnings

STYLE_KNOWLEDGE_BASE = {
    "streetwear": {
        "keywords": ["hoodie", "cargo", "oversized", "graphic tee", "streetwear", "skate", "jogger"],
        "forbidden_categories": ["Formal", "Tuxedos", "Suits & Blazers"],
        "parent_category": "Clothing"
    },
    "sporty": {
        "keywords": ["running", "activewear", "yoga", "breathable", "dry fit", "sneakers", "athletic"],
        "required_categories": ["Sports & Outdoors", "Athletic"],
        "parent_category": "Shoes"
    },
    "business_casual": {
        "keywords": ["blazer", "chino", "polo", "button-down", "loafers", "oxford", "office", "formal"],
        "forbidden_categories": ["Sweatpants", "Active"],
        "parent_category": "Clothing"
    },


    #new
    "casual": {
        "keywords": ["casual", "everyday", "relaxed fit", "basic tee", "weekend wear"],
        "forbidden_categories": [],
        "parent_category": "Clothing"
    },
    "vintage": {
        "keywords": ["vintage", "retro", "classic style", "distressed", "washed", "throwback"],
        "forbidden_categories": [],
        "parent_category": "Clothing"
    },
    "minimalist": {
        "keywords": ["minimalist", "simple", "solid color", "understated", "clean lines"],
        "forbidden_categories": [],
        "parent_category": "Clothing"
    }


}


EXPECTED_KEYS = {"style", "budget_max", "brands", "excluded_styles"}



def synthesize_cold_user_preferences(cold_user_ids: list, seed: int = 42) -> dict:
    """
    newly added tool functions are used to synthesize user preferences for cold-start users
    fixed random seed is adopted to ensure the reproducibility of the results
    this set of generation logic can be shared for both batch evaluation and user study
    """
    random.seed(seed)
    possible_styles = list(STYLE_KNOWLEDGE_BASE.keys())  #automatically follow the number of knowledge base styles
    possible_budgets = [25, 50, 75, 100, 150]
    possible_brands_options = [["Nike"], ["Adidas"], ["Nike", "Adidas"], []]

    return {
        uid: {
            "style": random.choice(possible_styles),
            "budget_max": random.choice(possible_budgets),
            "brands": random.choice(possible_brands_options)
        }
        for uid in cold_user_ids
    }






def safe_text_join(val): #convert all fields into searchable strings uniformly 
    if isinstance(val, (list, np.ndarray)):  #compatible with real Amazon ndarray corrupted data
        return ' '.join(str(x) for x in val)
    if pd.isna(val):
        return ''
    return str(val)


def refine_preferences(user_preferences: dict, feedback_tag: str) -> dict:  #dynamic fallback mechanism
    refined_prefs = user_preferences.copy()  #deep copy to prevent mutation

    if "excluded_styles" not in refined_prefs:
        refined_prefs["excluded_styles"] = []
    else:
        refined_prefs["excluded_styles"] = list(refined_prefs["excluded_styles"])

    if feedback_tag == "too_expensive":
        if "budget_max" in refined_prefs and refined_prefs["budget_max"] is not None:
            refined_prefs["budget_max"] *= 0.7

    elif feedback_tag == "not_my_style":
        current_style = refined_prefs.get("style")
        if current_style and current_style not in refined_prefs["excluded_styles"]:
            refined_prefs["excluded_styles"].append(current_style)

    return refined_prefs


"""
[core implementation logic]
precompute style_scores and blacklist matrices for all styles in STYLE_KNOWLEDGE_BASE.
runs only once globally to completely remove text retrieval complexity from the loop.

"""
def precompute_style_matrices(meta_df: pd.DataFrame) -> dict:
    print("[Performance Info] Executing off-line style rule precomputation...")
    title_desc = (
        meta_df['title'].apply(safe_text_join) + ' ' + meta_df['description'].apply(safe_text_join)
    ).str.lower()
    
    precomputed = {}
    
    for style, rules in STYLE_KNOWLEDGE_BASE.items():
        keywords = rules.get("keywords", [])
        forbidden = set(rules.get("forbidden_categories", []))
        required = set(rules.get("required_categories", []))  #    ！！Newly added reading whitelist
        #precompute cumulative style keyword scores
        score = pd.Series(0, index=meta_df.index, dtype=int)

        #precompute boolean masks for any style keyword
        keyword_match_mask = pd.Series(False, index=meta_df.index) #for fallback hard blacklist filtering
        
        for kw in keywords:
            has_kw = title_desc.str.contains(kw.lower(), regex=False)
            score += has_kw.astype(int)
            keyword_match_mask = keyword_match_mask | has_kw
            
        
        def is_category_forbidden(cat_array):  #category blacklist soft downgrading filter
            if isinstance(cat_array, (np.ndarray, list)):
                return not set(cat_array).isdisjoint(forbidden)
            return False
            
        forbidden_mask = meta_df['categories'].apply(is_category_forbidden)
        score = score.where(~forbidden_mask, 0) #zero out style scores for blacklisted categories,soft downgrading



        #if "required_categories", the score will only be retained if at least one of them is matched
        if required:
            def is_category_required_match(cat_array):
                if isinstance(cat_array, (np.ndarray, list)):
                    return not set(cat_array).isdisjoint(required)
                return False
            required_mask = meta_df['categories'].apply(is_category_required_match)
            score = score.where(required_mask, 0)  #not on the whitelist,reset to zero

        precomputed[style] = {
            "score": score,
            "keyword_match_mask": keyword_match_mask
        }

        
    print("[Performance Info] Precomputation matrices locked.")
    return precomputed


def recommend_knowledge_based(
    user_preferences: dict,
    meta_df: pd.DataFrame,
    top_n: int = 10,
    p95_price: float = None,
    precomputed_style_scores: dict = None,    #receive lookup table dictionary
    candidate_pool: list = None,   # ！！ new，the list of candidate product IDs for this user
) -> list:
    

    #候选池
    if candidate_pool is not None:
        df = meta_df[meta_df['parent_asin'].isin(candidate_pool)].copy()
    else:
        df = meta_df.copy()


    #input validation
    unexpected = set(user_preferences.keys()) - EXPECTED_KEYS
    if unexpected:
        warnings.warn(f"user_preferences contains unexpected: {unexpected}，will be ignored")


    """
    5-dimension high-performance lookup table sorting version
    personalized recommendation engine
    """


    
    #1,budget pre-screening, relaxed filtering
    budget_max = user_preferences.get("budget_max")
    if budget_max is not None:
        df = df[(df['price'] <= budget_max) | (df['price'].isna())]


    if df.empty:
        return []

    #2,dynamic truncation of extreme prices,reusing global budget values
    if p95_price is None:
        p95_price = meta_df['price'].quantile(0.95)
        if pd.isna(p95_price):
            p95_price = 35.99

    df['clipped_price'] = df['price'].clip(upper=p95_price).fillna(p95_price)


    #3, style expert lookup matching, completely avoiding text scanning
    style = user_preferences.get("style")
    excluded_styles = user_preferences.get("excluded_styles", [])

    # if style and style in STYLE_KNOWLEDGE_BASE and style not in excluded_styles and precomputed_style_scores:
    #     #direct index mapping lookup, achieving nanosecond-level memory speed
    #     df['style_score'] = precomputed_style_scores[style]["score"].reindex(df.index).fillna(0)
    # else:
    #     df['style_score'] = 0



    if style and style in STYLE_KNOWLEDGE_BASE and style not in excluded_styles:
        if precomputed_style_scores and style in precomputed_style_scores:
            #use precomputed lookup table for high speed
            df['style_score'] = precomputed_style_scores[style]["score"].reindex(df.index).fillna(0)


        else:
            #compute this single style on the fly without precomputed table, ensuring standalone call works correctly
            title_desc = (
                df['title'].apply(safe_text_join) + ' ' + df['description'].apply(safe_text_join)
            ).str.lower()
            rules = STYLE_KNOWLEDGE_BASE[style]
            keywords = rules.get("keywords", [])
            forbidden = set(rules.get("forbidden_categories", []))
            score = pd.Series(0, index=df.index)
            for kw in keywords:
                score += title_desc.str.contains(kw.lower(), regex=False).astype(int)
            def is_category_forbidden(cat_array):
                if isinstance(cat_array, (np.ndarray, list)):
                    return not set(cat_array).isdisjoint(forbidden)
                return False
            forbidden_mask = df['categories'].apply(is_category_forbidden)
            df['style_score'] = score.where(~forbidden_mask, 0)
    else:
        df['style_score'] = 0

    #4, brand preference score matching
    preferred_brands = user_preferences.get("brands", [])
    if preferred_brands:
        preferred_brands_lower = [b.lower() for b in preferred_brands]
        df['brand_score'] = df['store'].fillna('').str.lower().isin(preferred_brands_lower).astype(int)
    else:
        df['brand_score'] = 0

    #5,Bayesian smoothing,item quality smoothing score
    df['rating_number'] = df['rating_number'].fillna(0).astype(float)
    df['average_rating'] = df['average_rating'].fillna(0).astype(float)
    df['weighted_quality_score'] = df['average_rating'] * np.log1p(df['rating_number'])



    #6,price state indicator
    df['has_price_flag'] = df['price'].notna().astype(int)

    #7,5-dimension composite sorting
    df_sorted = df.sort_values(
        by=[
            'style_score',
            'brand_score',
            'has_price_flag',
            'clipped_price',
            'weighted_quality_score'
        ],
        ascending=[False, False, False, True, False]
    )

    ranked_items = df_sorted['parent_asin'].unique().tolist()
    if candidate_pool is not None:
        # 候选池模式下返回全部排序结果，不截断——截断交给 evaluate_model() 统一处理，
        # 跟 CF/Content-based/baseline 保持一致的约定
        return ranked_items
    return ranked_items[:top_n]


def get_knowledge_based_predictions_for_new_users(
    new_user_preferences: dict,
    meta_df: pd.DataFrame,
    top_n: int = 10,



   ####################### 每个用户的候选商品池
    candidate_pools: dict = None,
) -> dict:
    """
    Batch recommendation with three-level fallback. When candidate_pools is provided,
    each user is processed individually against their own candidate pool ,identical 
    preferences no longer share a cached result, since candidate pools differ per user.
    Fallback level 123 still searches the full catalog, not the candidate pool,
    see module docstring for why this hasn't been an issue in testing so far.
    """




    """
    batch recommendation + two-level smooth fallback mechanism (optimized de-duplication grouping version)
    core logic: users with identical (style, budget_max, brands, excluded_styles) share precomputed lists.
    each unique combination calculates recommendations only once and distributes to the group to prevent redundant computation.

    """


    batch_predictions = {}

    p95_price_global = meta_df['price'].quantile(0.95)
    if pd.isna(p95_price_global):
        p95_price_global = 35.99

    precomputed_style_scores = precompute_style_matrices(meta_df)

    meta_df_copy = meta_df.copy()
    meta_df_copy['rating_number'] = meta_df_copy['rating_number'].fillna(0).astype(float)
    meta_df_copy['average_rating'] = meta_df_copy['average_rating'].fillna(0).astype(float)
    meta_df_copy['global_quality'] = meta_df_copy['average_rating'] * np.log1p(meta_df_copy['rating_number'])





    #core optimization: deduplicate and group by preference combination
    from collections import defaultdict
    group_to_users = defaultdict(list)
    for user_id, prefs in new_user_preferences.items():
        key = (
            prefs.get("style"),
            prefs.get("budget_max"),
            tuple(sorted(prefs.get("brands", []))),
            tuple(sorted(prefs.get("excluded_styles", [])))
        )
        group_to_users[key].append(user_id)

    print(f"[info] {len(new_user_preferences)} users grouped into {len(group_to_users)} unique preference combinations (grouping is for organization only — candidate pools mean each user is still computed individually)")

    #compute only once per unique combination
    for key, user_ids in group_to_users.items():   #group by preference combination
        style, budget_max, brands_tuple, excluded_styles_tuple = key
        representative_prefs = {
            "style": style,
            "budget_max": budget_max,
            "brands": list(brands_tuple),
            "excluded_styles": list(excluded_styles_tuple)
        }

        for uid in user_ids:   #go through each user one on one
            user_pool = candidate_pools.get(uid) if candidate_pools else None

            recs = recommend_knowledge_based(
                representative_prefs, meta_df, top_n=top_n,
                p95_price=p95_price_global,
                precomputed_style_scores=precomputed_style_scores,
                candidate_pool=user_pool
            )

            excluded_styles = representative_prefs["excluded_styles"]
            hard_exclude_mask = pd.Series(False, index=meta_df.index)
            for ex_style in excluded_styles:
                if ex_style in precomputed_style_scores:
                    hard_exclude_mask = hard_exclude_mask | precomputed_style_scores[ex_style]["keyword_match_mask"]

            if excluded_styles:
                excluded_asin_set = set(meta_df.index[hard_exclude_mask].map(meta_df['parent_asin']))
                recs = [item for item in recs if item not in excluded_asin_set]



            #level 1,fallback
            if len(recs) < top_n:
                parent_cat = None
                if style and style in STYLE_KNOWLEDGE_BASE:
                    parent_cat = STYLE_KNOWLEDGE_BASE[style].get("parent_category")

                if parent_cat:
                    def inside_parent_category(cat_array):
                        if isinstance(cat_array, (np.ndarray, list)):
                            return parent_cat.lower() in [str(c).lower() for c in cat_array]
                        return False

                    cat_mask = meta_df_copy['categories'].apply(inside_parent_category)
                    cat_df = meta_df_copy[cat_mask & (~hard_exclude_mask)]

                    if budget_max is not None:
                        cat_df = cat_df[(cat_df['price'] <= budget_max) | (cat_df['price'].isna())]

                    if not cat_df.empty:
                        candidate_items = cat_df.sort_values(by=['global_quality'], ascending=False)['parent_asin'].tolist()
                        for item in candidate_items:
                            if item not in recs:
                                recs.append(item)
                            if len(recs) == top_n:
                                break

            #2
            if len(recs) < top_n:
                pop_df = meta_df_copy[~hard_exclude_mask]
                candidate_items = pop_df.sort_values(by=['global_quality'], ascending=False)['parent_asin'].tolist()
                for item in candidate_items:
                    if item not in recs:
                        recs.append(item)
                    if len(recs) == top_n:
                        break

            #3
            if len(recs) < top_n:
                candidate_items = meta_df_copy.sort_values(by=['global_quality'], ascending=False)['parent_asin'].tolist()
                for item in candidate_items:
                    if item not in recs:
                        recs.append(item)
                    if len(recs) == top_n:
                        break

            if candidate_pools is not None:
                # 候选池模式下返回全部排序结果，不截断
                final_recs = recs
            else:
                final_recs = recs[:top_n]

            #distribute single result to all users in this group
            batch_predictions[uid] = final_recs

    return batch_predictions



if __name__ == "__main__":
    print("Executing Danni's Precomputed Operational test loop...")

    mock_metadata = pd.DataFrame({
        'parent_asin': [f'B0000_ITEM_{i}' for i in range(1, 12)],
        'title': [
            'Oversized Urban Hype Hoodie Streetwear',
            'Men Leather Dress Shoes Formal',
            'Running Activewear Dynamic Shoes Gym',
            'Professional Work Office Blazer Blazer',
            'Cheap Casual Sandal',
            'Premium Diamond Watch Deluxe Edition',
            'Missing Price Socks Type A',
            'Missing Price Socks Type B',
            'Adidas Sport Socks Performance',
            'Nike Streetwear Cargo Pants Elite',
            'Standard Basic Shirt'
        ],
        'description': [np.array(['Hype streetwear fashion garment.'])] * 10 + [np.array([])],
        'categories': [
            np.array(['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Sweatshirts']),
            np.array(['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Suits & Blazers']),
            np.array(['Clothing, Shoes & Jewelry', 'Women', 'Shoes', 'Athletic']),
            np.array(['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Blazers']),
            np.array(['Clothing, Shoes & Jewelry', 'Unisex', 'Shoes']),
            np.array(['Clothing, Shoes & Jewelry', 'Luxury', 'Watches']),
            np.array(['Clothing, Shoes & Jewelry', 'Socks']),
            np.array(['Clothing, Shoes & Jewelry', 'Socks']),
            np.array(['Clothing, Shoes & Jewelry', 'Socks']),
            np.array(['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Pants']),
            np.array(['Clothing, Shoes & Jewelry', 'Basic'])
        ],
        'store': ['HypeBrand', 'ClassicTailor', 'Nike', 'OfficeWear', 'Generic', 'LuxuryCo', 'SocksCo', 'SocksCo', 'Adidas', 'Nike', 'BasicCo'],
        'price': [45.00, 120.00, 85.00, 95.00, 15.00, 3299.00, np.nan, np.nan, 12.00, 60.00, 25.00],
        'average_rating': [4.8, 4.9, 4.2, 4.5, 3.8, 5.0, 4.6, 4.2, 4.7, 4.9, 4.0],
        'rating_number': [1200, 10, 500, 80, 20, 1, 15, 300, 800, 2300, 50]
    })

    base_profile = {"style": "streetwear", "budget_max": 100.00, "brands": ["Nike"]}
    user_session_a = refine_preferences(base_profile, "not_my_style")
    user_session_b = refine_preferences(base_profile, "too_expensive")

    mock_batch_dict = {"user_cold_01": user_session_a, "user_cold_02": user_session_b}
    batch_results = get_knowledge_based_predictions_for_new_users(mock_batch_dict, mock_metadata, top_n=3)

    print("\nSelf-test on mock data completed successfully.")
    print("Note: full-scale real-data benchmarks (with real candidate pools) were run separately")
    print("during development — see knowledge_based_full_results.json for the actual output,")
    print("and report Method/Experiments section for performance benchmark history.")

