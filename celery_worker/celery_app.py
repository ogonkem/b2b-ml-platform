import os
from celery import Celery

REDIS_URL = f"redis://{os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', 6379)}"

celery_app = Celery(
    "b2b_ml_platform",
    broker=f"{REDIS_URL}/2",
    backend=f"{REDIS_URL}/3",
    include=["celery_worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
