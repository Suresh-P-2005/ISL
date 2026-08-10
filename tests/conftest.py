import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from config.testing import TestingConfig
from src.backend import create_app

@pytest.fixture
def app():
    return create_app(TestingConfig())

@pytest.fixture
def client(app):
    return TestClient(app)
