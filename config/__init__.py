import os
from .default import Config
from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig

def get_config():
    env = os.environ.get('FLASK_ENV', 'development').lower()
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    return DevelopmentConfig()
