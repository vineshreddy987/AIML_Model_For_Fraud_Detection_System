💳AIML Model for Fraud Detection System
A machine learning-driven system for detecting fraudulent financial transactions in real time. This project applies advanced classification models and anomaly detection techniques to analyze transaction patterns and flag suspicious activity with high accuracy.

🧠 Overview
This solution was developed for Pragyan’24, NIT Trichy as part of an intercollegiate technical challenge. The goal was to design an efficient and scalable fraud detection pipeline using Artificial Intelligence and Machine Learning techniques.

The project includes:

Data preprocessing and feature engineering

Evaluation of multiple machine learning models

Selection of the best-performing model

Readiness for integration with real-time systems

📊 Dataset
A synthetic dataset simulating financial transactions was used, featuring attributes such as:

type — Transaction category (TRANSFER, CASH_OUT, etc.)

amount — Amount transferred

oldbalanceOrg / newbalanceOrig — Sender's account balance

oldbalanceDest / newbalanceDest — Receiver's account balance

isFraud — Target label

📌 Dataset link: Fraud Detection Dataset (Replace with actual source)

⚙️ Technologies
Language: Python

Libraries: Scikit-learn, XGBoost, LightGBM, Pandas, NumPy

Visualization: Matplotlib, Seaborn

Version Control: Git, GitHub

(Deployment-ready backend planned using Flask)

🧩 Project Architecture
Raw Data
   ↓
Data Preprocessing & Cleaning
   ↓
Feature Engineering
   ↓
Model Training (Multiple Classifiers)
   ↓
Evaluation & Model Selection
   ↓
(Planned) Deployment
🛠 Model Development
Algorithms Evaluated:
Logistic Regression

Random Forest

XGBoost ✅ (Best performer)

LightGBM

Isolation Forest (for anomaly detection)

Feature Engineering:
Transaction behavior flags

Balance consistency checks

Transaction frequency metrics

✅ Results
Final Model: XGBoost
Evaluation Metrics:

Accuracy: 99.7%

Precision: 97.9%

Recall: 96.4%

AUC-ROC: 0.996

Optimized for imbalanced datasets using advanced tree-based ensemble methods.

🏃 Getting Started
# Clone the repository
git clone https://github.com/<your-username>/AIML_Model_For_Fraud_Detection_System.git
cd AIML_Model_For_Fraud_Detection_System

# Install dependencies
pip install -r requirements.txt

# Run the model
python fraud_detection_model.py

🔭 Future Work
Real-time integration with Flask/FastAPI

Model explainability with SHAP

Live monitoring dashboard using Streamlit

Continuous training pipeline

👨‍💻 Team
Team Name: SYNTAX SURFERS
Institution: Kalasalingam University
Event: Pragyan, NIT Trichy (Feb 21–22)
