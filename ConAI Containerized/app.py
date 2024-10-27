import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from flask import Flask, request, jsonify
from pdf2image import convert_from_path
from functools import wraps
import logging
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def validate_path(path):
    """Validate file path is safe and exists"""
    try:
        absolute_path = Path(path).resolve()
        allowed_directory = Path(CONFIG['data_directory']).resolve()
        return allowed_directory in absolute_path.parents and absolute_path.exists()
    except Exception:
        return False

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != CONFIG['api_key']:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/train', methods=['POST'])
@require_auth
def train():
    try:
        pdf_path = request.form.get('pdf_path')
        csv_path = request.form.get('csv_path')
        
        if not all([validate_path(p) for p in [pdf_path, csv_path]]):
            return jsonify({"error": "Invalid file paths"}), 400

        # Load and process data with error handling
        try:
            images, labels = load_data(pdf_path, csv_path)
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            return jsonify({"error": "Failed to load data"}), 500

        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        # Training with metrics collection
        model, history = train_model_with_monitoring(images, labels)
        
        # Save model with version control
        version = determine_model_version()
        model_path = f"models/model_v{version}.h5"
        model.save(model_path)
        
        return jsonify({
            "message": "Training completed",
            "model_path": model_path,
            "metrics": history,
            "version": version
        }), 200

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"error": "Training failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CONFIG['port'])
