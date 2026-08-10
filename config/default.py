import os
import secrets
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    MODELS_DIR = os.path.join(BASE_DIR, 'src', 'ml', 'artifacts', 'production')
    LEGACY_MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'real_landmark_data')
    VIDEO_DIR = os.path.join(BASE_DIR, 'video_landmark_data')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'src', 'web', 'templates', 'pages')
    STATIC_DIR = os.path.join(BASE_DIR, 'src', 'web', 'static')

    # ML Constants
    KEYFRAMES = 30
    N_FEAT = 126

    # Hand requirements mapping
    HAND_REQUIREMENTS = {
        "A": 2, "B": 2, "D": 2, "E": 2, "F": 2, "G": 2, "H": 2, 
        "J": 2, "K": 2, "M": 2, "N": 2, "P": 2, "Q": 2, "R": 2, 
        "S": 2, "T": 2, "W": 2, "X": 2, "Y": 2, "Z": 2
    }

    ALL_WORDS = [
        "Hello","Yes","No","I Love You","Help","Thank You","Sorry","Please",
        "Good","Bad","More","Finished","Again","Understand","Mother","Father",
        "Sister","Brother","Baby","Friend","Family","Person","Water","Food",
        "Sleep","Eat","Drink","Tired","Sick","Medicine","Toilet","Home",
        "Stop","Wait","Danger","Call","Police","Fire","Pain","Come",
        "Where","What","Who","When","Which","Why"
    ]
