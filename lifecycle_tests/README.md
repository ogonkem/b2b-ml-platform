# lifecycle_tests/

Drives the real, running app through every stage of the product story —
train, single predict, bulk predict, drift detection, weekly retrain — using
the non-overlapping holdout slices in `notebooks/archive/holdout/`
(generate them first with `python notebooks/make_holdout_splits.py` if that
folder doesn't exist yet).

This is separate from `tests/`: those are pytest unit/integration suites;
these are standalone scripts that exercise the live docker-compose stack end
to end, including real MLflow promotions and real Airflow DAG runs — not
mocked. Run them from a bash shell with the repo's `.venv` python.

## Prerequisites

```bash
docker compose up -d          # full stack must be running
python notebooks/make_holdout_splits.py   # if notebooks/archive/holdout/ doesn't exist
```

## Running

Each stage is runnable on its own once its prerequisites are met (each
script checks for what it needs in `.state.json` / holdout files and exits
with a clear message if something upstream hasn't run yet):

```bash
python lifecycle_tests/01_train_champion.py
python lifecycle_tests/02_single_predict.py
python lifecycle_tests/03_bulk_predict.py
python lifecycle_tests/04_labeled_data_stable.py
python lifecycle_tests/05_labeled_data_drift.py
python lifecycle_tests/06_weekly_retrain.py
python lifecycle_tests/07_verify_hotswap.py
```

Or run the whole thing in order:

```bash
bash lifecycle_tests/run_all.sh
```

## Stage order and what each one does

1. **Train champion** — trains on `train_slice.csv` only (never touches the
   other holdout rows), registers + promotes it to Production directly,
   bypassing the normal promotion gate (which needs 2 runs to compare and
   can't bootstrap a first model). Sets the baseline everything else scores
   against.
2. **Single predict** — scores `single_predict_holdout.csv` through
   `/v1/predict`, confirms `/health` reports the baseline version.
3. **Bulk predict** — restarts `celery_worker` (it caches the model at
   startup and never reloads), uploads a 900-row subset of
   `bulk_predict_holdout.csv` to `/v1/batch/upload` (a single request can't
   exceed the 1,000-row/month tenant quota — see Known constraints below),
   scores the downloaded results.
4. **Labeled data — stable** (negative control) — uploads
   `labeled_data_stable.csv`, triggers the real `selastone_daily_ingestion`
   DAG, confirms it does *not* commit.
5. **Labeled data — drift** — uploads `labeled_data_drift.csv` (engineered
   low-Credit_Score rows), triggers the same DAG, confirms it *does* commit
   for real (author: `selastone-mlops-bot`).
6. **Weekly retrain** — triggers the real `selastone_weekly_retrain` DAG
   (`sync_data -> train_model -> promote_model`), reads back both MLflow
   runs, reports whether the real 2pp-AUC gate promoted a challenger.
7. **Verify hotswap** — confirms `/health` and live `/v1/predict` calls agree
   on whatever stage 6 concluded — re-validates the `ModelManager` hot-swap
   path.

## Known constraints or side effects to expect

- **Stage 3's 900-row cap**: `check_and_increment_quota` enforces a
  1,000-row/month cap per tenant token (`app/main.py`). Re-running stage 3
  more than once in the same calendar month will hit `429 Quota exceeded`
  once the token's monthly total passes 1,000 — reset with
  `docker exec data-redis redis-cli DEL "quota:<token>:<YYYY_MM>"` for local
  testing (matches whatever `API_TOKENS`'s first token is in `.env`).
- **Stage 5 makes a real git commit** on whatever branch is checked out,
  authored by `selastone-mlops-bot <mlops-bot@selastone.local>` — this is
  the intended, approved behavior, not a bug.
- **Stage 6 can take several minutes** — `train_model` runs the full
  `notebooks/retrain.py` (4 models, SMOTE, tuned hyperparameters) inside the
  Airflow container against the full baseline + merged feedback data.
- Re-running stage 4/5 clears the `labeled-data` MinIO bucket first — local
  dev only, don't point this at anything with real tenant data in it.
- **Re-running the full suite twice in a row can fail stage 5** with `git
  commit` returning exit 1 ("nothing to commit") — `labeled_data_drift.csv`
  is a static fixture, so a second run produces a byte-identical
  `feedback_labeled.csv`/`.dvc` pointer to the one already committed, and
  there's genuinely nothing new to commit. Expected git behavior, not a bug;
  in real operation each day's labeled data differs. If you need a fully
  clean re-run, `git rm notebooks/archive/feedback_labeled.csv.dvc` (or just
  accept stage 1 will train on the previously-committed feedback too — it
  merges unconditionally, same as the real weekly DAG).
