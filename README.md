# 🐄 Cattle Unified Classifier

This project is a unified Flask web application that combines multiple AI/ML models for cattle disease detection.

---

## 📁 Project Structure
cattle_unified_classifier/
│
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
│
├── models/ # Model files (not included in GitHub)
│ ├── cattle_3class_classifier.keras
│ ├── footrot_mobilenet_final_model.keras
│ ├── cattle_udder_mobilenet_model.h5
│ ├── tongue_classification_mobilenetv2.h5
│ ├── footrot_class_indices.json
│ └── tongue_model_config.json
│
├── templates/
│ └── index.html # Unified interface
│
└── static/
├── css/style.css
└── js/script.js

yaml
Copy code

---

## 🚀 How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/cattle_unified_classifier.git
   cd cattle_unified_classifier
Create a virtual environment and install dependencies

bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
Run the Flask app

bash
Copy code
python app.py
Open your browser and go to http://127.0.0.1:5000

💾 Model Files
Model files (.keras, .h5) are not uploaded to GitHub due to large size.
Download them separately and place inside the models/ folder.

🧠 Included Models
Cattle 3-Class Disease Classifier

Footrot Detector (MobileNet)

Udder Health Classifier (MobileNet)

Tongue Disease Classifier (MobileNetV2)

📜 License
This project is for educational and research purposes only.
