# B2B Machine Learning Risk Scoring Platform 🚀

A comprehensive, enterprise-grade machine learning platform built entirely with open-source software. This repository contains the architecture, infrastructure blueprints, and core service frameworks designed to run locally on a laptop using Docker Desktop and python virtual environments.

The platform provides secure, authenticated real-time prediction workflows (<300ms p95 latency) alongside a decoupled asynchronous batch pipeline capable of handling large client datasets smoothly.

## 🛠️ System Architecture & Open-Source Stack

This platform replaces costly cloud-native dependencies with highly optimized open-source packages:

*   **Ingestion & Routing:** `HAProxy` at the edge for secure SSL/TLS termination and path routing.
*   **Live Core Serving:** `FastAPI` powered by an optimized vector processing pipeline via `scikit-learn` and live memory execution via `XGBoost`/`LightGBM`.
*   **Asynchronous Bulk Pipeline:** Distributed task worker infrastructure managed by `Celery` using `Redis OSS` as a task broker.
*   **Storage & Caching Layers:** `MinIO` for S3-compatible local object storage, `PostgreSQL` for operational metadata management, and `Redis OSS` for rate-limiting.
*   **MLOps Retraining Loop:** Data version control with `DVC`, experimental asset tracking and deployment lifecycle management via `MLflow Server & Registry`, and orchestration automated by `Apache Airflow`.
*   **Zero-Downtime Deployment:** Custom hot-reloading daemon utilizing `Redis Streams` to hot-swap active memory references without dropping connection pools.
*   **Telemetry & Observability:** Real-time logging funneled into a column-oriented `ClickHouse OSS` database, application instrumentation scraped by `Prometheus`, and end-to-end visualization via `Grafana` dashboards featuring tracking for Population Stability Index (PSI) data drift.

## 📁 Repository Layout

├── .github/               # CI/CD action workflows
├── app/                   # Real-time FastAPI serving application
│   ├── main.py            # API gateway routes & validation hooks
│   └── model_manager.py   # Hot-reload daemon & thread-safe model swapper
├── shared/                # Shared internal utilities (Parity Layer)
│   └── features.py        # Feature engineering pipeline & missing value imputers
├── airflow-dags/          # Scheduled workflow configuration files
├── docker-data/           # (Ignored) Local physical storage mount for containers
├── tests/                 # Automated unit, integration, and system test suites
├── docker-compose.yml     # Master local cloud container infrastructure blueprint
└── README.md              # Project documentation
