# B2B ML Risk Scoring Platform

A self-contained machine learning platform for credit risk scoring built entirely on open-source tooling. The full stack — API, workers, model registry, scheduler, and observability — runs on a single laptop with Docker Compose.

Two workloads are supported:

- **Real-time scoring** — authenticated REST requests return a default probability in under 300 ms
- **Bulk batch scoring** — CSV files of any size are queued asynchronously, scored in the background, and made available via a presigned download link

An automated MLOps loop keeps the model current: tenants upload labeled outcomes, a daily job measures data drift, and a weekly pipeline retrains and promotes a new model if the challenger beats the incumbent by a meaningful margin.

---

## Contents

1. [Data storage layer](#1-data-storage-layer)
2. [Feature engineering](#2-feature-engineering)
3. [Model training](#3-model-training)
4. [Drift detection](#4-drift-detection)
5. [Model registry and data versioning](#5-model-registry-and-data-versioning)
6. [Prediction API](#6-prediction-api)
7. [Async batch pipeline](#7-async-batch-pipeline)
8. [MLOps automation](#8-mlops-automation)
9. [Observability](#9-observability)
10. [Running locally](#running-locally)
11. [Repository layout](#repository-layout)

---

## 1. Data storage layer

Everything else in this platform depends on four data stores, each chosen for a specific job.

### PostgreSQL

Airflow uses Postgres to store DAG run history, task states, and scheduler metadata. It is the memory of the orchestration layer — without it, the scheduler cannot survive a restart.

> **Closed-source alternative:** Amazon RDS or Google Cloud SQL manage backups, failover, and patching automatically. For teams already on AWS, Aurora Postgres adds automatic read-replica scaling.

### Redis

Redis serves two separate roles:

**Message broker for the batch pipeline.** When a CSV is uploaded, the API drops a task message onto a Redis queue. The Celery worker picks it up and processes it independently. Two Redis databases are used (`/2` for the broker, `/3` for the result backend) so message traffic and result storage don't interfere with each other.

**Per-tenant quota counters.** Each tenant's monthly prediction count is a Redis key (`quota:{tenant}:{year_month}`). The `INCRBY` command is atomic — concurrent requests cannot double-count. A 32-day TTL on the key means cleanup is automatic. Using a relational database for this would require a scheduled cleanup job and be noticeably slower under load.

> **Closed-source alternative:** AWS SQS replaces Redis as a broker with durable queues and dead-letter handling managed for you. AWS API Gateway handles rate limiting before traffic hits your application. RabbitMQ is the open-source broker with the richest routing features if task priorities or dead-letter exchanges are needed.

### MinIO

MinIO is an S3-compatible object store that runs in a container. It holds four buckets:

| Bucket | Contents |
|---|---|
| `raw-landing` | CSV files uploaded by tenants before scoring |
| `batch-results` | Scored CSVs ready for tenant download |
| `labeled-data` | Tenant-uploaded ground truth for drift detection and retraining |
| `mlflow-artifacts` | Serialised models and run metadata written by MLflow |

When a batch job completes, the API generates a presigned URL — the tenant downloads the results file directly from MinIO without the API acting as a proxy. This keeps the API stateless and removes a large data transfer from its critical path.

> **Closed-source alternative:** AWS S3, Google Cloud Storage, or Azure Blob Storage — all speak the same S3 protocol, so swapping `MINIO_HOST` and credentials is the only required change.

### ClickHouse

Every prediction is written to a `prediction_logs` table in ClickHouse. ClickHouse stores data column by column rather than row by row, which makes range aggregations — "average default probability per tenant across all of last month" — very fast. This suits audit and monitoring queries well. A second table, `labeled_uploads`, records when tenants submit actuals, forming a lightweight chain-of-custody trail.

> **Closed-source alternative:** BigQuery, Snowflake, and Redshift are managed columnar databases that remove the operational overhead of running ClickHouse. Elasticsearch is common for logs but performs poorly on numeric aggregations. A regular Postgres table works at small volumes but degrades as prediction rows accumulate into the millions.

---

## 2. Feature engineering

**`shared/features.py` — `FeaturePipeline`**

The `FeaturePipeline` class is the preprocessing contract shared between training and serving. It learns from training data (`.fit()`) and applies exactly the same transformations to new requests (`.transform()`). Keeping it in `shared/` means both the training script and the API import from one place — a divergence in the preprocessing logic is one of the most common silent failures in ML systems.

What it does at fit time:
- Identifies columns with more than 40% missing values and marks them for removal
- Saves the **median** of every numeric column and the **mode** of every categorical column (medians are more robust than means in financial data, where a handful of very large loans can skew the average substantially)

What it does at transform time:
- Fills missing numeric values with saved medians, missing categoricals with saved modes
- Creates three ratio features that carry more signal than the raw amounts:
  - `loan_to_income` — loan size relative to monthly income
  - `loan_to_property` — how much of the property value is being financed (a proxy for loan-to-value)
  - `credit_to_income` — creditworthiness relative to income

`fit()` explicitly excludes the three derived feature names from `self.numeric_cols` even if they are present in the training DataFrame. This matters because `retrain.py` computes derived features before building the training matrix, so they already exist when `FeaturePipeline.fit(X_train)` is called. Without the exclusion, `transform()` would find them in `numeric_cols` and append them again — producing 35 columns when the model expects 32. The scaler is fit on the real 32-column training data, and `self.feature_names` stores the authoritative column order so `transform()` can `reindex()` to exactly match training layout regardless of what order columns arrive in at inference time.

> **Alternative:** `sklearn.pipeline.Pipeline` bundles preprocessing and model into a single serialisable object, which eliminates this class of version-skew issue by construction. The tradeoff is less visibility into individual steps and more friction when debugging preprocessing logic.

---

## 3. Model training

**`notebooks/retrain.py`**

The training script runs headlessly — no plots, no interactive cells, no manual steps. It is designed to be called by Airflow on a schedule or by a developer from the command line.

**The training sequence:**

1. **Load** the DVC-tracked baseline CSV (148,670 rows, 34 columns). If a `feedback_labeled.csv` exists — written by the daily drift DAG when tenant actuals triggered a retraining signal — it is merged in before any splitting.

2. **Clean** — drop high-missing columns, impute numeric columns with medians, encode categoricals as integers with `LabelEncoder`.

3. **Engineer features** — create the three ratio features described above.

4. **Remove leaky features** — drop `Interest_rate_spread`, `rate_of_interest`, and `Upfront_charges`. These are only observable at loan closing, not at origination. Keeping them produces unrealistically high accuracy at training time and fails silently in production — a model that "knows" the interest rate it ended up charging is effectively seeing the answer.

5. **Scale** — fit a `StandardScaler` on the resulting 32-column training set and save it as `scaler.pkl`.

6. **SMOTE** — the dataset is class-imbalanced (~75% non-default, ~25% default). SMOTE synthesises new minority-class examples in feature space rather than duplicating existing rows, pushing the training distribution toward 44% defaults.

7. **Train four candidates** — XGBoost, LightGBM, Logistic Regression, Random Forest. XGBoost and LightGBM use hyperparameters from `best_hyperparams.json`, produced by a prior `RandomizedSearchCV` run. Re-tuning every week would be slow and introduce noise from random seeds.

8. **Compare and log** — all four runs are logged to MLflow with full metrics. The best AUC winner is saved to `notebooks/models/best_model.pkl`.

**Why these four models?**

Gradient boosting (XGBoost / LightGBM) consistently outperforms on tabular credit data — it handles mixed feature types, missing values, and non-linear interactions without manual feature engineering. LightGBM is typically faster at training; XGBoost is often slightly more accurate on smaller datasets.

Logistic Regression is the interpretability baseline. If a regulator asks why a specific application was declined, logistic coefficients are directly explainable in a way that boosted trees are not. SHAP values bridge this gap for the tree models (`shap_background.pkl` is saved for use by the API's explainability endpoint).

Random Forest anchors the comparison — it has lower variance than a single tree and shows how much of the boost comes from gradient descent versus ensemble averaging alone.

**Why SMOTE over class weighting?**

Class weighting is simpler and adjusts only the loss function. SMOTE changes the actual training distribution, which can help tree ensembles find better decision boundaries in the minority region since splits are based on feature thresholds, not loss gradients. Both are valid. SMOTE adds training time; class weights add a hyperparameter that can interact unexpectedly with `scale_pos_weight` in XGBoost.

> **Closed-source alternative:** AWS SageMaker Training Jobs manage compute provisioning, spot instance interruption, and artefact storage. Vertex AI (GCP) and Azure ML offer equivalent managed training pipelines. This platform trades those managed conveniences for full local reproducibility and no cloud spend during development.

---

## 4. Drift detection

**`shared/drift.py` — `compute_psi` / `check_drift`**

Population Stability Index (PSI) measures how much a feature's distribution has shifted between the training baseline and incoming tenant actuals.

```
PSI < 0.10   →  stable, no action
0.10 – 0.20  →  moderate shift, worth monitoring
≥ 0.20       →  significant drift, trigger retraining
```

The calculation splits both distributions into histogram buckets and compares the percentage of values in each bucket. A small epsilon prevents division-by-zero when a bucket is empty in one of the distributions.

PSI is computed across seven key numeric features: `loan_amount`, `property_value`, `income`, `Credit_Score`, `LTV`, `dtir1`, `term`. If any one of them crosses 0.2, the daily ingestion DAG marks the data as needing a retraining run.

**Why PSI over a statistical test like KS or chi-squared?**

The KS test gives a p-value but not a magnitude — you know the distributions differ, but not by how much. PSI gives both direction and severity in a single number and has a well-established industry threshold (0.2) that credit risk practitioners recognise from Basel II monitoring guidelines. It is also symmetric: shifts in both tails are treated equally.

> **Alternative:** `evidently` and `whylogs` are purpose-built drift monitoring libraries that produce richer reports and integrate with dashboards out of the box. They add a dependency; `shared/drift.py` has none beyond NumPy and pandas.

---

## 5. Model registry and data versioning

### MLflow (port 5000)

MLflow handles two jobs:

**Experiment tracking.** Every training run logs its hyperparameters, evaluation metrics (AUC, F1, accuracy, precision, recall), and the serialised model to MinIO. The UI at `http://localhost:5000` shows all runs side by side and lets you compare models across retraining cycles.

**Model registry.** The winning model from each run is registered as a version of `selastone_credit_scorer`. Versions move through lifecycle stages (`Staging → Production`). The API's `ModelManager` polls the registry every 60 seconds and replaces the in-memory model when a new Production version appears — no restart required.

### DVC

Model artefacts live in MLflow; training *data* is versioned with DVC. DVC commits tiny pointer files (`.dvc`) to git while the actual CSVs sit in MinIO. This means git history records exactly which data produced each model without storing large files in the repository.

> **Closed-source alternative:** Weights & Biases (W&B) and Neptune.ai are commercial alternatives to MLflow with richer visualisation and team collaboration features. SageMaker Model Registry manages versioning and deployment together. MLflow is the only fully open-source option with a built-in UI, REST API, and Python SDK. For data versioning, Pachyderm and Delta Lake offer stronger lineage guarantees than DVC at the cost of more infrastructure.

---

## 6. Prediction API

**`app/main.py` — FastAPI (port 8000)**

The API is the system's front door. It handles authentication, quota enforcement, real-time scoring, batch ingestion, and labeled data collection.

### Authentication

Every request must include `Authorization: Bearer <token>`. Valid tokens are loaded from the `API_TOKENS` environment variable (comma-separated). Lookup is O(1) against a Python `set` regardless of tenant count. The token doubles as the tenant identifier — it appears in MinIO object paths, Redis quota keys, and ClickHouse logs, so a single credential ties all activity together without a separate session layer.

> **Closed-source alternative:** OAuth2 with JWTs allows token expiry, fine-grained scopes, and refresh without redeployment. AWS API Gateway and GCP Cloud Endpoints handle authentication before traffic reaches the application server.

### Real-time scoring — `POST /v1/predict`

1. Validate the 32-field loan application payload with Pydantic
2. Atomically increment the tenant's monthly quota counter in Redis; raise HTTP 429 if exceeded
3. Impute missing values, create derived features, scale to the 32-column training order
4. Call `model.predict_proba()` — returns a default probability and a binary decision
5. Write the prediction event to ClickHouse
6. Record request latency and outcome in Prometheus counters

### Batch upload — `POST /v1/batch/upload`

1. Parse and validate the CSV
2. Deduct the row count from the tenant's monthly quota in one `INCRBY` call (bulk jobs consume quota proportionally to size)
3. Upload the file to MinIO `raw-landing/{tenant}/{date}/{job_id}.csv`
4. Write `job:{job_id}:status = queued` to Redis
5. Enqueue a Celery task and return the `job_id` immediately — the HTTP response is not blocked on scoring

### Labeled data — `POST /v1/labeled-data`

Tenants upload CSVs with actual loan outcomes (`actual_outcome`: 0 or 1). These land in MinIO `labeled-data/` and feed the daily drift check. Requiring an outcome column at upload time enforces data quality before it enters the pipeline.

### Model hot-swap — `app/model_manager.py`

`ModelManager` runs a background daemon thread that polls MLflow every 60 seconds. When a new Production version is detected:

1. The new model is downloaded from MinIO **outside** the lock — this can take several seconds
2. The in-memory pointer is swapped **inside** a `threading.Lock()` — this takes microseconds

Prediction threads hold the lock only during the pointer swap, not during the download. The result is that serving is never blocked on a model download. In-flight requests during the swap window continue using the previous model; the next request after the swap uses the new one.

> **Alternative:** Blue/green deployment at the infrastructure level — two API instances behind a load balancer, traffic switched atomically — achieves zero-downtime rollouts without any in-process threading. Kubernetes rolling updates provide this with a single field change in the deployment manifest.

---

## 7. Async batch pipeline

**`celery_worker/tasks.py` — Celery**

Celery is a distributed task queue. A large CSV could take minutes to score — processing it synchronously inside the HTTP request would hold a connection open and block other requests. Instead, the API enqueues a task and returns a `job_id` in milliseconds.

The Celery worker runs the same Docker image as the API (sharing all model dependencies). For each batch job it:

1. Downloads the raw CSV from MinIO `raw-landing`
2. Applies the same imputation, derived features, scaling, and 32-column alignment as the real-time path — vectorised across the entire DataFrame at once rather than row by row
3. Calls `model.predict_proba()` on the full matrix — one forward pass scores the whole file
4. Writes a results CSV to MinIO `batch-results/{tenant}/{date}/{job_id}_results.csv`
5. Sets `job:{job_id}:status = complete` in Redis and stores the result object path

The client polls `GET /v1/batch/results/{job_id}`. When the job is complete, the API generates a presigned MinIO URL (valid for one hour) so the client downloads the file directly from object storage rather than through the API.

**Job status is tracked in plain Redis keys** rather than in Celery's result backend. This means the API can query job status with a single `GET` without going through Celery's result API, and status keys automatically expire after 24 hours via TTL.

> **Closed-source alternative:** AWS SQS + Lambda achieves the same pattern without managing worker containers — Lambda functions scale to zero when there are no jobs and scale out automatically under load. For very high throughput with replay requirements, Kafka provides durable, ordered queues. RabbitMQ adds priority queues and dead-letter exchanges if sophisticated task routing is needed.

---

## 8. MLOps automation

**`airflow_dags/` — Apache Airflow (port 8080)**

Two DAGs automate the model lifecycle. Tasks within a DAG pass data to each other via XCom — Airflow's cross-task communication mechanism — avoiding shared files between steps.

### Daily ingestion DAG (`selastone_daily_ingestion`, 01:00 UTC)

```
pull_labeled_data → check_psi_drift → commit_to_dvc
```

**`pull_labeled_data`** downloads every CSV from MinIO `labeled-data`, concatenates them, and saves a combined file. It passes the row count downstream via XCom.

**`check_psi_drift`** loads the DVC-tracked baseline CSV and the combined labeled pull, then runs PSI across seven numeric features. If fewer than 100 labeled rows are available the task exits early — PSI on tiny samples is statistically meaningless.

**`commit_to_dvc`** only executes when drift ≥ 0.2. It saves the merged labeled data to `notebooks/archive/feedback_labeled.csv`, runs `dvc add` and `dvc push` to upload the file to MinIO, and commits the `.dvc` pointer to git. This pointer is what the weekly retrain DAG pulls down to include the new data.

### Weekly retraining DAG (`selastone_weekly_retrain`, Monday 02:00 UTC)

```
sync_data → train_model → promote_model
```

**`sync_data`** runs `git pull` followed by `dvc pull` to fetch any new `.dvc` pointer files and download the corresponding data from MinIO. This is how the labeled feedback from the daily DAG enters the training run.

**`train_model`** calls `notebooks/retrain.py`. All four candidate models are trained and logged to MLflow.

**`promote_model`** compares the two most recent runs in the experiment by test AUC. The challenger is registered as a new version of `selastone_credit_scorer` and promoted to Production only if it improves AUC by at least 2 percentage points. A smaller improvement could reflect random seed variance or overfitting on the test split rather than a genuinely better model.

> **Closed-source alternative:** Prefect and Dagster offer more modern Python-native DAG APIs with better type safety and native async. AWS Step Functions, GCP Workflows, and Azure Data Factory replace Airflow with fully managed schedulers. MLflow Projects can run training scripts without a separate orchestrator if the workflow is simple enough.

---

## 9. Observability

**Prometheus (port 9090) + Grafana (port 3000)**

The API exposes a `/metrics` endpoint that Prometheus scrapes every 15 seconds. Metrics tracked:

| Metric | Labels | What it measures |
|---|---|---|
| `predictions_total` | `tenant_id`, `prediction` | Count of 0/1 decisions per tenant |
| `prediction_latency_seconds` | — | End-to-end request latency histogram |
| `batch_uploads_total` | `tenant_id`, `status` | Upload outcomes (success / invalid / quota_exceeded / minio_error) |
| `batch_rows_total` | `tenant_id` | Total rows processed per tenant per month |

Grafana dashboards show these alongside PSI drift scores queried from ClickHouse. A rising PSI trend on the dashboard is an early warning before the formal 0.2 threshold triggers retraining — it gives operators time to investigate whether drift reflects a genuine distribution shift or a data quality issue upstream.

> **Closed-source alternative:** Datadog, New Relic, and Dynatrace provide full-stack observability with managed infrastructure and alerting. For ML-specific monitoring, Arize AI and Fiddler monitor prediction distributions and model performance degradation with less setup than building custom dashboards.

---

## Running locally

**Requirements:** Docker Desktop (8 GB RAM allocated), Python 3.12

```bash
# 1. Start the full stack
git clone <repo-url>
cd b2b-ml-platform
docker compose up -d

# 2. Train the first model (runs inside the stack on its own network)
docker run --rm \
  --network b2b-ml-platform_backend-network \
  -v "$(pwd)/notebooks:/app/notebooks" \
  -v "$(pwd)/shared:/app/shared:ro" \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  -e AWS_ACCESS_KEY_ID=minio_admin_user \
  -e AWS_SECRET_ACCESS_KEY=super_secure_minio_pass_2026 \
  b2b-ml-platform-api \
  sh -c "pip install -q imbalanced-learn && python notebooks/retrain.py"

# 3. Run the integration test suite
pip install mlflow httpx pytest pytest-asyncio python-dotenv
python -m pytest tests/integration/ -v
```

**Service endpoints:**

| Service | URL |
|---|---|
| Prediction API + Swagger docs | http://localhost:8000/docs |
| MLflow experiment tracker | http://localhost:5000 |
| MinIO console | http://localhost:9001 |
| Airflow scheduler | http://localhost:8080 |
| Grafana dashboards | http://localhost:3000 |
| Prometheus metrics | http://localhost:9090 |

---

## Repository layout

```
├── app/
│   ├── main.py              # FastAPI: /v1/predict, /v1/batch/upload, /v1/labeled-data
│   └── model_manager.py     # Polling hot-swap daemon — threading.Lock + MLflow registry
├── shared/
│   ├── features.py          # FeaturePipeline: fit/transform, imputation, derived features
│   └── drift.py             # PSI: compute_psi(), check_drift()
├── celery_worker/
│   ├── celery_app.py        # Celery config — Redis db/2 broker, db/3 backend
│   └── tasks.py             # process_batch: MinIO read → score → MinIO write → Redis status
├── airflow_dags/
│   ├── ingestion.py         # Daily DAG: pull labeled CSVs → PSI check → DVC commit
│   ├── weekly_retrain.py    # Weekly DAG: git pull → retrain → promote
│   └── promotion.py         # promote_if_better(): 2% AUC gate before Production swap
├── notebooks/
│   ├── retrain.py           # Headless training: 4 models → MLflow logging → artefact save
│   └── models/              # Artefacts: best_model.pkl, scaler.pkl, feature_names.json …
├── tests/
│   ├── unit/                # Isolated unit tests — all external calls mocked
│   └── integration/         # Live stack tests: API endpoints, batch pipeline, MLops loop
├── docker-compose.yml       # 10-service stack on a shared bridge network
├── Dockerfile.api           # Python 3.12-slim — shared image for API and Celery worker
└── requirements.txt         # Runtime deps: fastapi, celery, xgboost, lightgbm, mlflow …
```
