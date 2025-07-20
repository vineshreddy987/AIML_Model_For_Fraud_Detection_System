# 💳 AIML-Powered Fraud Detection System

FraudGuard is an AI-powered fraud detection system designed to identify suspicious financial transactions using advanced machine learning techniques. Built for precision and scalability, the system evaluates transactional patterns and flags potential fraud in real time — with a focus on performance, security, and interpretability.

---

## ✨ Features

🔍 Real-time fraud detection using AI/ML  
📊 Preprocessing and feature engineering for high signal extraction  
🤖 Multiple ML models evaluated (XGBoost, LightGBM, Isolation Forest)  
📈 Handles imbalanced data using advanced ensemble methods  
🛡️ Ready for deployment with clean and modular codebase  
📁 Detailed result metrics: accuracy, precision, recall, F1, AUC-ROC  

---

## 🧱 Tech Stack

| Layer           | Technology                           |
|----------------|---------------------------------------|
| Language        | Python 3.8+                           |
| ML Libraries    | Scikit-learn, XGBoost, LightGBM       |
| Data Handling   | Pandas, NumPy                         |
| Visualization   | Matplotlib, Seaborn                   |
| Environment     | Python Virtualenv                     |
| Version Control | Git, GitHub                           |
| Deployment (Planned) | Flask API Backend                 |

---
## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- pip
- Git

### 🛠 Installation Steps

```bash
# Clone the repository
git clone https://github.com/vineshreddy987/AIML_Model_For_Fraud_Detection_System.git

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the main script
python fraud_detection_model.py
```
---
📊 **Model Summary**
| Model               | Precision    | Recall    | F1-Score  | AUC-ROC   |
| ------------------- | ------------ | --------- | --------- | --------- |
| Logistic Regression | 92.3%        | 89.4%     | 90.8%     | 0.947     |
| Random Forest       | 96.7%        | 94.1%     | 95.4%     | 0.981     |
| **XGBoost** ✅       | **97.9%**    | **96.4%** | **97.1%** | **0.996** |
| LightGBM            | 97.4%        | 95.2%     | 96.3%     | 0.991     |
| Isolation Forest    | Unsupervised | N/A       | N/A       | 0.823     |

---
**🔭 Future Enhancements**

🌐 Real-time prediction using Flask or FastAPI

📊 Interactive dashboard with Streamlit

🧠 Model explainability via SHAP and LIME

🧹 Automated retraining pipeline using CI/CD
🔒 Integration with authentication layers for enterprise use
---
**👥 Team**
Team Name: SYNTAX SURFERS,
Institution: Kalasalingam University,
Event: Pragyan, NIT Tiruchirappalli (Feb 21–22).
