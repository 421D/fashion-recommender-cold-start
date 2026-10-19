"""Run reproducible CF-SVD ranking experiments on the real project data.

By default this trains on the complete train split and evaluates a fixed random
sample of 5,000 users from the shared candidate pool. Use ``--all-users`` only
when a full sampled-ranking evaluation is required; it can take several hours.
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import pandas as pd

from collaborative_filtering_model import (
    get_cf_rating_predictions,
    recommend_cf,
    train_svd,
)
from evaluation import evaluate_model


DEFAULT_TRAIN_PATH = "processed_data/Clothing_Shoes_and_Jewelry_train.parquet"
DEFAULT_TEST_PATH = "processed_data/Clothing_Shoes_and_Jewelry_test.parquet"
DEFAULT_CANDIDATES_PATH = "candidate_pools.json"
DEFAULT_GROUND_TRUTH_PATH = "ground_truth.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CF-SVD and evaluate it against the shared candidate pools."
    )
    user_group = parser.add_mutually_exclusive_group()
    user_group.add_argument(
        "--max-users",
        type=int,
        default=5_000,
        help="Number of randomly sampled evaluation users (default: 5000).",
    )
    user_group.add_argument(
        "--all-users",
        action="store_true",
        help="Evaluate every user in ground_truth.json; this can take hours.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-factors", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--include-rating-metrics",
        action="store_true",
        help="Also calculate RMSE/MAE on the selected users' test interactions.",
    )
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--candidate-pools-path", default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--ground-truth-path", default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON file to save the resulting metrics.",
    )
    return parser.parse_args()


def choose_users(ground_truth, max_users, all_users, seed):
    available_users = sorted(str(user_id) for user_id in ground_truth)
    if all_users or max_users >= len(available_users):
        return available_users
    if max_users <= 0:
        raise ValueError("--max-users must be positive")
    return random.Random(seed).sample(available_users, max_users)


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    for path_text in (
        args.train_path,
        args.candidate_pools_path,
        args.ground_truth_path,
    ):
        if not Path(path_text).is_file():
            raise FileNotFoundError(f"Required input file not found: {path_text}")

    print("Loading shared evaluation files...")
    with open(args.candidate_pools_path, encoding="utf-8") as file:
        candidate_pools = json.load(file)
    with open(args.ground_truth_path, encoding="utf-8") as file:
        ground_truth = json.load(file)

    target_users = choose_users(
        ground_truth, args.max_users, args.all_users, args.seed
    )
    missing_pools = [user_id for user_id in target_users if user_id not in candidate_pools]
    if missing_pools:
        raise ValueError(
            f"candidate_pools is missing {len(missing_pools)} selected users; "
            f"first missing user: {missing_pools[0]}"
        )
    selected_ground_truth = {user_id: ground_truth[user_id] for user_id in target_users}
    print(f"Selected {len(target_users):,} users for evaluation.")

    print("Loading train split and training SVD...")
    train_df = pd.read_parquet(args.train_path)
    train_started = time.perf_counter()
    model = train_svd(
        train_df,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        random_state=args.seed,
    )
    train_seconds = time.perf_counter() - train_started
    print(f"SVD training finished in {train_seconds / 60:.1f} minutes.")

    rating_predictions = None
    if args.include_rating_metrics:
        if not Path(args.test_path).is_file():
            raise FileNotFoundError(f"Test split not found: {args.test_path}")
        print("Calculating RMSE/MAE for the selected users...")
        test_df = pd.read_parquet(args.test_path)
        selected_test = test_df[test_df["user_id"].astype(str).isin(target_users)]
        rating_predictions = get_cf_rating_predictions(model, selected_test)
        del test_df, selected_test
        gc.collect()

    # The model has already learned from train_df; retaining the DataFrame only
    # wastes memory during the  user-by-candidate ranking loop.
    del train_df
    gc.collect()

    print("Ranking candidates...")
    rank_started = time.perf_counter()
    predictions = {}
    for index, user_id in enumerate(target_users, start=1):
        predictions[user_id] = recommend_cf(
            model,
            user_id,
            candidate_pools[user_id],
            top_n=len(candidate_pools[user_id]),
        )
        if index % args.progress_every == 0 or index == len(target_users):
            elapsed = time.perf_counter() - rank_started
            print(f"  Ranked {index:,}/{len(target_users):,} users in {elapsed / 60:.1f} minutes")

    metrics = evaluate_model(
        "CF-SVD",
        predictions,
        selected_ground_truth,
        k=args.top_k,
        rating_predictions=rating_predictions,
    )
    metrics.update(
        {
            "evaluated_users": len(target_users),
            "n_factors": args.n_factors,
            "n_epochs": args.n_epochs,
            "seed": args.seed,
            "training_seconds": round(train_seconds, 2),
            "ranking_seconds": round(time.perf_counter() - rank_started, 2),
        }
    )

    print("\nCF-SVD evaluation results")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        print(f"Saved metrics to {args.output_path}")


if __name__ == "__main__":
    main()
