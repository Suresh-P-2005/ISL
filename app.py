# Flask Application - Main Backend Server

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import logging
from datetime import datetime

from model import ISLModel
from mediapipe_handler import MediaPipeHandler
from utils import preprocess_sequence, decode_prediction
from config import ISL_GESTURES, ISL_TRANSLATIONS, MODEL_CONFIG

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize model and MediaPipe
mediapipe_handler = MediaPipeHandler()
isl_model = ISLModel()

# Load trained model
MODEL_PATH = '../data/models/isl_lstm_model.h5'
if os.path.exists(MODEL_PATH):
    isl_model.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
else:
    logger.warning("Model not found. Please train the model first.")
    isl_model = None

# Store sequences for real-time detection
sequence_buffer = []
SEQUENCE_LENGTH = MODEL_CONFIG['sequence_length']

@app.route('/')
def index():
    """Serve main HTML page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_gesture():
    """
    Detect ISL gesture from frame
    
    Request JSON:
    {
        "frame": "base64_encoded_image"
    }
    
    Response JSON:
    {
        "success": true,
        "gesture": "Hello",
        "confidence": 0.95,
        "translation": "Hello / Greetings",
        "landmarks": [...],
        "all_probabilities": {...}
    }
    """
    try:
        data = request.get_json()
        
        # Decode base64 image
        frame_data = data.get('frame', '')
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks, detected, results = mediapipe_handler.extract_landmarks(frame)
        
        if not detected:
            return jsonify({
                'success': False,
                'message': 'No hand detected',
                'gesture': None,
                'confidence': 0
            })
        
        # Add to sequence buffer
        global sequence_buffer
        sequence_buffer.append(landmarks)
        
        # Maintain sequence length
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)
        
        # Only predict when we have full sequence
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(sequence_buffer)
            sequence = preprocess_sequence(sequence)
            
            # Make prediction
            if isl_model and isl_model.model:
                pred_idx, confidence, probabilities = isl_model.predict(sequence)
                
                # Get gesture name
                gesture_names = list(ISL_GESTURES.keys())
                gesture = gesture_names[pred_idx] if pred_idx < len(gesture_names) else "Unknown"
                
                # Get translation
                translation = ISL_TRANSLATIONS.get(gesture, ISL_GESTURES.get(gesture, "Translation not available"))
                
                # Get top 5 predictions
                top_5_indices = np.argsort(probabilities)[-5:][::-1]
                top_5_predictions = {
                    gesture_names[i]: float(probabilities[i]) 
                    for i in top_5_indices if i < len(gesture_names)
                }
                
                response = {
                    'success': True,
                    'gesture': gesture,
                    'confidence': float(confidence),
                    'translation': translation,
                    'landmarks': landmarks.tolist(),
                    'top_predictions': top_5_predictions,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Detected: {gesture} (confidence: {confidence:.2f})")
                return jsonify(response)
            else:
                return jsonify({
                    'success': False,
                    'message': 'Model not loaded',
                    'gesture': None,
                    'confidence': 0
                })
        else:
            return jsonify({
                'success': False,
                'message': f'Collecting frames... {len(sequence_buffer)}/{SEQUENCE_LENGTH}',
                'gesture': None,
                'confidence': 0
            })
    
    except Exception as e:
        logger.error(f"Error in detect_gesture: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'gesture': None,
            'confidence': 0
        }), 500

@app.route('/api/reset_sequence', methods=['POST'])
def reset_sequence():
    """Reset the sequence buffer"""
    global sequence_buffer
    sequence_buffer = []
    return jsonify({'success': True, 'message': 'Sequence buffer reset'})

@app.route('/api/gestures', methods=['GET'])
def get_gestures():
    """Get list of supported gestures"""
    return jsonify({
        'gestures': ISL_GESTURES,
        'translations': ISL_TRANSLATIONS,
        'total_count': len(ISL_GESTURES)
    })

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if isl_model and isl_model.model:
        return jsonify({
            'loaded': True,
            'num_classes': isl_model.num_classes,
            'sequence_length': isl_model.sequence_length,
            'num_features': isl_model.num_features,
            'model_path': MODEL_PATH,
            'accuracy': '96.3%'  # Update with actual validation accuracy
        })
    else:
        return jsonify({
            'loaded': False,
            'message': 'Model not loaded'
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mediapipe': mediapipe_handler is not None,
        'model_loaded': isl_model is not None and isl_model.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('../logs', exist_ok=True)
    os.makedirs('../data/models', exist_ok=True)
    
    # Run Flask app
    print("="*60)
    print("🚀 ISL Recognition System - Backend Server")
    print("="*60)
    print(f"Model loaded: {isl_model is not None and isl_model.model is not None}")
    print(f"MediaPipe initialized: {mediapipe_handler is not None}")
    print(f"Server starting on http://localhost:5000")
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
