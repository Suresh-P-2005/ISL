from .default import Config

class TestingConfig(Config):
    DEBUG = False
    TESTING = True
