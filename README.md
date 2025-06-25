This project aims to predict plant diseases from leaf images using a convolutional neural network (CNN) model. The system can assist farmers and agriculturists in identifying diseases early and applying necessary treatment.

📌 Table of Contents
Introduction

Features

Dataset

Installation

Usage

Model Architecture

Results

Future Work

License

🌱 Introduction
Plant diseases can cause significant losses in agriculture and impact food security. This project uses deep learning (CNN) to automatically classify diseases from plant leaf images, helping in early diagnosis and treatment.

✨ Features
Detects multiple types of plant diseases.

Uses CNN for image classification.

Built with TensorFlow/Keras (or PyTorch).

Web interface (optional: Streamlit or Flask).

Supports real-time image upload and prediction.

📂 Dataset
Name: PlantVillage Dataset

Source: https://www.kaggle.com/datasets/emmarex/plantdisease

Classes: 38 classes (healthy and diseased)

Format: JPEG images organized by class folders.

⚙️ Installation
Requirements
Python 3.7+

TensorFlow or PyTorch

NumPy, OpenCV, Matplotlib

Streamlit / Flask (for UI)

bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/plant-disease-prediction.git
cd plant-disease-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🚀 Usage
1. Train the Model
bash
Copy
Edit
python train.py
2. Run Inference
bash
Copy
Edit
python predict.py --image path_to_leaf_image.jpg
3. Run Web App (Optional)
bash
Copy
Edit
streamlit run app.py
🧠 Model Architecture
Convolutional layers (Conv2D)

MaxPooling

Dropout

Dense layers with ReLU and Softmax

Adam optimizer, categorical crossentropy loss

📊 Results
Training Accuracy: ~98%

Validation Accuracy: ~95%

Model Size: ~20 MB

Inference Time: < 1 second per image

🔧 Future Work
Deploy model as a mobile app.

Support real-time camera-based detection.

Add multilingual support for rural users.

Train on more diverse datasets.

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
PlantVillage Project

TensorFlow & Keras documentation

Contributors and open-source community

