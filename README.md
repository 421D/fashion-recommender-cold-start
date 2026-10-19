# Fashion Recommender with a Cold-Start Path

A hybrid Top-10 recommender for fashion e-commerce, built on 8.5 million real Amazon reviews.
Users with a purchase history get a fusion of collaborative filtering, content-based and popularity
rankings. Brand-new users with no history get a knowledge-based recommender driven by four onboarding
answers: style, budget, preferred brands and styles to avoid.

Built as a group project by a team of five.

**Stack:** Python · pandas · NumPy · scikit-learn (TF-IDF, cosine similarity) · Surprise (SVD) · PyArrow

---

## My role: knowledge-based cold-start module (Danni Cai)

I designed and built [`src/knowledge_based_model.py`](src/knowledge_based_model.py), which serves
**every cold-start user** in the hybrid system.

**Ranking.** After a relaxed budget filter (items without a price are kept, not dropped), items are
ranked on five keys in order:

1. Style match, from a hand-built style knowledge base of keywords and forbidden categories
2. Preferred brand
3. Has a listed price (about 16% of items have none; they rank lower instead of being removed)
4. Price, lower first, capped at the 95th percentile (about $83) so luxury outliers don't distort it
5. Quality: average rating weighted by the log of the rating count

**Feedback loop.** "Too expensive" lowers the budget by 30%; "not my style" adds the style to the
excluded list, and the batch path hard-excludes items matching it.

**Making it fast enough.** Ranking 28,252 cold-start users against 294,021 items was estimated at
**13.2 hours**:

| Step | Runtime |
|---|---|
| Naive per-user ranking | 13.2 h (estimated) |
| Precompute a style-score matrix once, offline | 1.8 h |
| Group users by preference combination (only about 60 unique combinations) | **~15 s** |
| Final evaluation on per-user candidate pools (27,573 users) | 5.19 min |

That is a speed-up of more than 3,000× before candidate-pool evaluation was added.

---

## How the system works

```
                 ┌─ has history? ──── yes ──► CF-SVD ───────┐
User ──► Switch ─┤                            Content-based ├─► Reciprocal Rank Fusion ─► Top-10
                 │                            Popularity ───┘    (0.3 / 0.3 / 0.4)
                 └─ no ───────────────────► Knowledge-based ──────────────────────────► Top-10
```

| Module | File | What it does |
|---|---|---|
| Data preparation | [`data_preparation.py`](src/data_preparation.py) | Downloads Amazon Reviews 2023 (Clothing, Shoes & Jewelry), keeps 2020 onwards, iterative 5-core filter, 80/10/10 temporal split, tags warm and cold test users |
| Collaborative filtering | [`collaborative_filtering_model.py`](src/collaborative_filtering_model.py), [`run_cf_evaluation.py`](src/run_cf_evaluation.py) | Surprise SVD on ratings |
| Content-based | [`content_based_model.py`](src/content_based_model.py) | TF-IDF over title, description, features, categories and store. Recommends **complementary** items (a jacket leads to trousers or shoes, not another jacket) using category pairs mined from same-user purchases within 7 days, with curated rules as fallback |
| Knowledge-based | [`knowledge_based_model.py`](src/knowledge_based_model.py) | Cold-start recommender described above |
| Hybrid | [`hybrid_model.py`](src/hybrid_model.py) | Switching by user type, then weighted Reciprocal Rank Fusion, because the three scores live on different scales |
| Evaluation | [`evaluation.py`](src/evaluation.py), [`generate_candidate_pools.py`](src/generate_candidate_pools.py) | Shared protocol: every method ranks the same per-user pool (held-out items + 99 random negatives, seed 42) |

---

## Results

Offline, Top-10, same candidate pools for every method:

| Method | Precision@10 | Recall@10 | NDCG@10 | Diversity | Novelty |
|---|---|---|---|---|---|
| Random | 0.0263 | 0.0991 | 0.0585 | 0.9799 | 19.19 |
| Category Popularity | 0.0899 | 0.4124 | 0.2559 | 0.9696 | 16.81 |
| Global Popularity | 0.0930 | 0.4426 | 0.3176 | 0.9752 | 15.81 |
| **Knowledge-based** | 0.0496 | 0.0896 | 0.0740 | 0.9728 | **19.21** |
| Content-based | 0.0552 | 0.2594 | 0.1755 | 0.9526 | 19.06 |
| CF-SVD | 0.0582 | 0.2758 | 0.1953 | 0.9792 | 17.29 |
| **Hybrid** | 0.0724 | **0.3747** | **0.2457** | 0.9738 | 16.97 |

User study (10 participants, real preferences, Knowledge-based and Hybrid): CTR@10 0.42,
conversion@10 0.15, and satisfaction correlated with clicks and purchases (r = 0.64 to 0.68).

How to read these numbers:

- **Popularity wins on offline accuracy** because a few items take most interactions (long tail).
  Hybrid is the strongest *personalised* method and keeps more novelty than popularity.
- **Knowledge-based recall is at the random baseline offline.** Its input was synthetic onboarding
  preferences, which don't reflect why people actually bought things. It has the highest novelty of all
  seven methods, and in the user study with real preferences it performed much better. That study has
  only 10 people, so treat it as a direction, not proof.

---

## Running it

```bash
pip install -r requirements.txt     # Python 3.10
python src/data_preparation.py      # downloads the public dataset, writes processed_data/
python src/generate_candidate_pools.py
python src/run_cf_evaluation.py     # CF-SVD on a 5,000-user sample (--all-users for full)
```

Each model file also has a self-test on small mock data: `python src/knowledge_based_model.py`.

The dataset is public ([Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab) and is
downloaded by the script, not stored in this repo. Generated predictions and candidate pools (several
hundred MB) are not included either.

## Team

| Member | Module |
|---|---|
| **Danni Cai** | **Knowledge-based cold-start recommender** |
| Liuting | Data preparation and exploratory analysis |
| Junwei | Collaborative filtering (SVD) and content-based recommender |
| Yuxuan | Hybrid model and experiment design |
| Chuliang | Shared evaluation module and user study |

## Limitations

- Knowledge-based offline evaluation used synthetic preferences; the user study is small (n = 10) and self-reported.
- CF-SVD is limited by data sparsity (5 latent factors, 2 epochs to keep training manageable).
- TF-IDF captures shared text, not visual style.
- The baseline and user-study scripts are not included in this repo.
