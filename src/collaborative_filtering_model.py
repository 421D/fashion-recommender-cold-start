"""
collaborative_filtering_model.py

Collaborative filtering baseline/model using Surprise SVD.

Input data format from data_preparation.py:
    columns: user_id, parent_asin, rating, timestamp, date

Public recommendation format:
    {user_id: [item_id_1, item_id_2, ..., item_id_N], ...}
"""

from typing import Iterable, Optional

import pandas as pd


REQUIRED_COLUMNS = {"user_id", "parent_asin", "rating"}


def _validate_reviews_df(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _to_raw_id(value) -> str:
    """Surprise is most predictable when category ids are converted to strings."""
    return str(value)


def _get_trainset(model):
    trainset = getattr(model, "trainset", None)
    if trainset is None:
        raise ValueError("The SVD model has no trainset. Train it with train_svd() first.")
    return trainset


def _all_train_items(model) -> list:
    trainset = _get_trainset(model)
    return [trainset.to_raw_iid(inner_iid) for inner_iid in trainset.all_items()]


def _seen_items_for_user(model, user_id: str) -> set:
    trainset = _get_trainset(model)
    raw_user_id = _to_raw_id(user_id)
    try:
        inner_uid = trainset.to_inner_uid(raw_user_id)
    except ValueError:
        return set()
    return {trainset.to_raw_iid(inner_iid) for inner_iid, _ in trainset.ur[inner_uid]}


def train_svd(train_df: pd.DataFrame, n_factors: int = 5, n_epochs: int = 2, **kwargs):
    """
    Train a Surprise SVD model.

    Parameters
    ----------
    train_df:
        DataFrame with at least user_id, parent_asin, rating.
    n_factors:
        Number of latent factors.
    n_epochs:
        Number of SGD epochs.
    **kwargs:
        Passed to surprise.SVD. Useful options include lr_all, reg_all,
        biased, random_state, verbose, min_rating, and max_rating.

    Returns
    -------
    surprise.prediction_algorithms.matrix_factorization.SVD
        A fitted Surprise SVD model.
    """
    try:
        from surprise import Dataset, Reader, SVD
    except ImportError as exc:
        raise ImportError(
            "scikit-surprise is required for collaborative filtering. "
            "Install it with: pip install scikit-surprise"
        ) from exc

    _validate_reviews_df(train_df, "train_df")

    ratings_df = train_df[["user_id", "parent_asin", "rating"]].copy()
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["parent_asin"] = ratings_df["parent_asin"].astype(str)
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
    ratings_df = ratings_df.dropna(subset=["rating"])

    if ratings_df.empty:
        raise ValueError("train_df contains no valid ratings after cleaning.")

    min_rating = float(kwargs.pop("min_rating", 1.0))
    max_rating = float(kwargs.pop("max_rating", 5.0))
    random_state = kwargs.pop("random_state", 42)
    verbose = kwargs.pop("verbose", False)

    reader = Reader(rating_scale=(min_rating, max_rating))
    dataset = Dataset.load_from_df(ratings_df, reader)
    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )
    model.fit(trainset)
    return model


def predict_rating(model, user_id: str, item_id: str) -> float:
    """
    Predict a user's rating for one item.

    This is mainly for CF RMSE/MAE evaluation. Surprise can still return an
    estimate for unknown users/items by falling back to learned/global biases.
    """
    pred = model.predict(_to_raw_id(user_id), _to_raw_id(item_id))
    return float(pred.est)


def get_cf_rating_predictions(model, test_df: pd.DataFrame) -> tuple:
    """
    Batch-predict ratings on the test set for RMSE/MAE.

    Returns
    -------
    tuple[list[float], list[float]]
        y_true and y_pred with matching order.
    """
    _validate_reviews_df(test_df, "test_df")

    y_true = []
    y_pred = []
    for row in test_df[["user_id", "parent_asin", "rating"]].itertuples(index=False):
        user_id, item_id, rating = row
        if pd.isna(rating):
            continue
        y_true.append(float(rating))
        y_pred.append(predict_rating(model, user_id, item_id))
    return y_true, y_pred


def recommend_cf(model, user_id: str, candidate_items: list, top_n: int = 10) -> list:
    """
    Generate Top-N recommendations for a single user.

    The returned list is sorted by predicted rating from high to low. Items
    already seen by the user in the training set are filtered out.
    """
    if top_n <= 0:
        return []

    seen_items = _seen_items_for_user(model, user_id)
    scored_items = []
    for item_id in candidate_items:
        raw_item_id = _to_raw_id(item_id)
        if raw_item_id in seen_items:
            continue
        score = predict_rating(model, user_id, raw_item_id)
        scored_items.append((raw_item_id, score))

    scored_items.sort(key=lambda item_score: item_score[1], reverse=True)
    return [item_id for item_id, _ in scored_items[:top_n]]


def get_cf_predictions_for_all_users(
    model,
    test_df: pd.DataFrame,
    top_n: int = 10,
    candidate_items: Optional[Iterable[str]] = None,
    user_ids: Optional[Iterable[str]] = None,
) -> dict:
    """
    Generate Top-N recommendation lists for all target users.

    Parameters
    ----------
    model:
        Fitted Surprise SVD model.
    test_df:
        Used to infer the target test users when user_ids is not provided.
    top_n:
        Number of recommendations per user.
    candidate_items:
        Optional item universe to rank. If omitted, all items seen during
        training are used. For large-scale experiments, pass a smaller
        candidate set sampled consistently for all models.
    user_ids:
        Optional target users. If omitted, all unique users in test_df are used.

    Returns
    -------
    dict
        {user_id: [item_id_1, item_id_2, ...]}
    """
    if "user_id" not in test_df.columns:
        raise ValueError("test_df is missing required column: user_id")

    if candidate_items is None:
        candidate_items = _all_train_items(model)
    else:
        candidate_items = [_to_raw_id(item_id) for item_id in candidate_items]

    if user_ids is None:
        user_ids = test_df["user_id"].dropna().astype(str).unique().tolist()
    else:
        user_ids = [_to_raw_id(user_id) for user_id in user_ids]

    predictions = {}
    for user_id in user_ids:
        predictions[user_id] = recommend_cf(model, user_id, candidate_items, top_n=top_n)
    return predictions

def get_cf_predictions_from_candidate_pools(model, candidate_pools: dict) -> dict:
    """
    Generate CF predictions using per-user candidate pools (from
    evaluation.build_candidate_pools). Unlike get_cf_predictions_for_all_users,
    this returns every item in each pool fully ranked, not just top-10 --
    truncation to K is handled inside evaluate_model() so all methods are
    compared on the exact same footing.

    Parameters
    ----------
    model:
        Fitted Surprise SVD model.
    candidate_pools:
        {user_id: [candidate_item_ids]}, from evaluation.build_candidate_pools().

    Returns
    -------
    dict
        {user_id: [all candidate items, ranked by predicted rating, high to low]}
    """
    predictions = {}
    for user_id, candidates in candidate_pools.items():
        predictions[user_id] = recommend_cf(model, user_id, candidates, top_n=len(candidates))
    return predictions


if __name__ == "__main__":
    # Small smoke test that does not require the full Amazon dataset.
    demo_train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "parent_asin": ["i1", "i2", "i1", "i3", "i2"],
            "rating": [5, 3, 4, 5, 4],
        }
    )
    demo_test = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u4"],
            "parent_asin": ["i3", "i2", "i1"],
            "rating": [4, 3, 5],
        }
    )

    svd_model = train_svd(demo_train, n_factors=5, n_epochs=5)
    y_true_demo, y_pred_demo = get_cf_rating_predictions(svd_model, demo_test)
    print("rating predictions:", list(zip(y_true_demo, y_pred_demo)))
    print("top-n predictions:", get_cf_predictions_for_all_users(svd_model, demo_test, top_n=2))
