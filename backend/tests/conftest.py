import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://muse:muse_password@localhost:5432/muse",
)

os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client