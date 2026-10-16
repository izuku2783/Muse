import os

import pytest
from fastapi.testclient import TestClient


os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "false"


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
