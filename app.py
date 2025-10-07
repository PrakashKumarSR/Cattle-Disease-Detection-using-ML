import os
import logging
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'unified-cattle-classifier-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global variables for models
models = {}
model_configs = {}

# Model configurations
MODEL_INFO = {
    'lumpy_skin': {
        'name': 'Cattle Disease Classifier',
        'path': 'models/cattle_3class_classifier.keras',
        'type': 'keras',
        'classes': ['Lumpy Skin Disease', 'Not Cattle', 'Healthy Cow'],
        'input_size': (224, 224),
        'description': 'Detects Lumpy Skin Disease in cattle'
    },
    'footrot': {
        'name': 'Footrot (FMD) Classifier',
        'path': 'models/footrot_mobilenet_final_model.keras',
        'type': 'keras',
        'classes_file': 'models/footrot_class_indices.json',
        'input_size': (224, 224),
        'description': 'Detects Foot and Mouth Disease in cattle feet'
    },
    'udder': {
        'name': 'Udder Health Classifier',
        'path': 'models/cattle_udder_mobilenet_model.h5',
        'type': 'h5',
        'classes': ['NON CATTLE IMAGES', 'mastitis teats', 'normal teats'],
        'input_size': (224, 224),
        'description': 'Detects mastitis in cattle udders'
    },
    'tongue': {
        'name': 'Tongue Disease Classifier',
        'path': 'models/tongue_classification_mobilenetv2.h5',
        'type': 'h5',
        'classes_file': 'models/tongue_model_config.json',
        'input_size': (224, 224),
        'description': 'Detects tongue diseases in cattle'
    }
}

def load_all_models():
    """Load all models at startup"""
    global models, model_configs
    
    for model_key, info in MODEL_INFO.items():
        try:
            # Check if model file exists
            if not os.path.exists(info['path']):
                logger.warning(f"⚠️ Model not found: {info['path']}")
                continue
            
            # Load model
            logger.info(f"🔄 Loading {info['name']}...")
            model = tf.keras.models.load_model(info['path'], compile=False)
            
            # Recompile
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            models[model_key] = model
            
            # Load class mapping
            if 'classes_file' in info:
                if os.path.exists(info['classes_file']):
                    with open(info['classes_file'], 'r') as f:
                        class_data = json.load(f)
                        
                    # Handle different JSON structures
                    if 'class_names' in class_data:
                        model_configs[model_key] = class_data['class_names']
                    elif 'class_indices' in class_data:
                        # Reverse mapping from index to class name
                        indices = class_data['class_indices']
                        model_configs[model_key] = [None] * len(indices)
                        for class_name, idx in indices.items():
                            model_configs[model_key][idx] = class_name
                    else:
                        # Assume it's direct class_indices
                        model_configs[model_key] = [None] * len(class_data)
                        for class_name, idx in class_data.items():
                            model_configs[model_key][idx] = class_name
                else:
                    logger.warning(f"⚠️ Class file not found: {info['classes_file']}")
                    model_configs[model_key] = info.get('classes', [])
            else:
                model_configs[model_key] = info['classes']
            
            logger.info(f"✅ {info['name']} loaded successfully!")
            logger.info(f"   Classes: {model_configs[model_key]}")
            
        except Exception as e:
            logger.error(f"❌ Error loading {info['name']}: {e}")
            continue
    
    logger.info(f"\n🎉 Loaded {len(models)}/{len(MODEL_INFO)} models successfully!")
    return len(models) > 0

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def get_result_details(model_key, predicted_class, confidence):
    """Get detailed results based on model and prediction"""
    
    # Common patterns
    is_non_cattle = any(x in predicted_class.lower() for x in ['non cattle', 'not cattle', 'non_cattle'])
    is_healthy = 'healthy' in predicted_class.lower() or 'normal' in predicted_class.lower()
    
    if is_non_cattle:
        return {
            'status': 'WARNING',
            'message': '⚠️ This does not appear to be a valid cattle image',
            'recommendation': 'Please upload a clear image of cattle in the appropriate body part',
            'color': '#f59e0b',
            'icon': '🚫',
            'risk_level': 'N/A'
        }
    elif is_healthy:
        return {
            'status': 'HEALTHY',
            'message': '✅ The cattle appears healthy',
            'recommendation': 'No immediate action needed. Continue regular monitoring.',
            'color': '#22c55e',
            'icon': '✅',
            'risk_level': 'LOW'
        }
    else:
        # Disease detected
        disease_recommendations = {
            'lumpy skin disease': 'URGENT: Isolate the animal and consult a veterinarian immediately. Lumpy Skin Disease is highly contagious.',
            'fmd': 'URGENT: Quarantine the animal and contact a veterinarian. Foot and Mouth Disease spreads rapidly.',
            'footrot': 'URGENT: Isolate and treat immediately. Clean and dry the affected area. Veterinary consultation recommended.',
            'mastitis': 'URGENT: Milk affected quarters separately. Apply prescribed treatment. Consult veterinarian for antibiotic therapy.',
            'diseased': 'URGENT: Isolate the animal and seek immediate veterinary consultation.'
        }
        
        recommendation = 'URGENT: Consult a veterinarian immediately!'
        for key, rec in disease_recommendations.items():
            if key in predicted_class.lower():
                recommendation = rec
                break
        
        return {
            'status': 'DISEASE',
            'message': f'🚨 Possible {predicted_class} detected',
            'recommendation': recommendation,
            'color': '#ef4444',
            'icon': '🚨',
            'risk_level': 'HIGH'
        }

@app.route('/')
def index():
    """Render main page"""
    available_models = {}
    for key, info in MODEL_INFO.items():
        available_models[key] = {
            'name': info['name'],
            'description': info['description'],
            'loaded': key in models
        }
    
    return render_template('index.html', 
                         models=available_models,
                         total_loaded=len(models))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get selected model
        model_key = request.form.get('model_type')
        
        if not model_key or model_key not in models:
            return jsonify({'error': 'Invalid or unavailable model selected'}), 400
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process image
        image = Image.open(file.stream)
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Preprocess for prediction
        model_info = MODEL_INFO[model_key]
        processed_image = preprocess_image(image, model_info['input_size'])
        
        # Make prediction
        model = models[model_key]
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class name
        class_names = model_configs[model_key]
        predicted_class = class_names[predicted_class_idx]
        
        # Get all probabilities
        probabilities = {}
        for i, class_name in enumerate(class_names):
            probabilities[class_name] = float(predictions[0][i])
        
        # Get detailed results
        details = get_result_details(model_key, predicted_class, confidence)
        
        # Prepare response
        result = {
            'success': True,
            'model_name': model_info['name'],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percent': f"{(confidence * 100):.2f}%",
            'probabilities': probabilities,
            'image': f"data:image/jpeg;base64,{img_str}",
            **details
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'total_models': len(MODEL_INFO),
        'available_models': list(models.keys())
    })

# Initialize the app
if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("🐄 UNIFIED CATTLE CLASSIFICATION SYSTEM")
    print("=" * 60)
    
    # Load all models
    if load_all_models():
        print("\n✅ System ready!")
    else:
        print("\n⚠️ No models loaded. Please check model files.")
    
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)