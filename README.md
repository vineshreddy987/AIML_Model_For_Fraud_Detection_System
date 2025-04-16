# AIML_Model_For_Fraud_Detection_System
Team No : 18
Team Name : SYNTAX SURFERS
Kalasalingam University 
Pragyan, NIT Tiruchirappalli
Feb 21 – 22

Fraud Detection System
Project Overview
This project aims to build a fraud detection system that identifies fraudulent transactions in real-time. The system uses machine learning models to analyze transaction data and predict whether a transaction is fraudulent or legitimate. The project includes data preprocessing, feature engineering, model training, evaluation, and deployment.

Dataset
The dataset used for this project contains transaction details, including:

Transaction Type: Type of transaction (e.g., CASH_OUT, PAYMENT, TRANSFER).

Amount: Transaction amount.

Original Balance: Balance before the transaction.

New Balance (Origin): Balance after the transaction.

Destination Old Balance: Destination account balance before the transaction.

Destination New Balance: Destination account balance after the transaction.

Dataset Link
You can find the dataset here: Fraud Detection Dataset

Models Used
The following machine learning models were trained and evaluated for fraud detection:

Logistic Regression

Random Forest

XGBoost

LightGBM

Isolation Forest (Anomaly Detection)

Best Implementation
The XGBoost model was selected as the best-performing model due to its high accuracy and ability to handle imbalanced data. The implementation steps include:

Data Preprocessing:

Handle missing values.

Encode categorical variables (e.g., transaction type).

Normalize numerical features.

Feature Engineering:

Create new features (e.g., transaction frequency, balance changes).

Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

Model Training:

Split the data into training and testing sets (80-20 split).

Train the XGBoost model on the training set.

Model Evaluation:

Evaluate the model on the testing set using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.

Model Deployment:

Deploy the model using a Flask API for real-time predictions.

Model Accuracy and Metrics
The XGBoost model achieved the following performance metrics:

Metric	Value
Accuracy	99.93%
Precision	66.14%
Recall	99.88%
F1 Score	79.58%
AUC-ROC	1.0000
Interpretation of Metrics:
High Accuracy (99.93%): The model performs well overall.

High Recall (99.88%): The model is excellent at identifying fraudulent transactions.

Moderate Precision (66.14%): The model has a relatively high false positive rate, which can be improved.

High AUC-ROC (1.0000): The model has perfect discriminatory power.

How to Run the Project
Clone the Repository:

bash
Copy
git clone https://github.com/vineshreddy987/AIML_Model_For_Fraud_Detection_System.git
cd fraud-detection
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Run the Flask App:

bash
Copy
python app.py
Access the Web Interface:

Open your browser and go to http://127.0.0.1:5000/.

Submit transaction details to get real-time fraud predictions.

Future Improvements
Improve Precision:

Adjust the classification threshold to reduce false positives.

Use cost-sensitive learning to prioritize fraud detection.

Handle Concept Drift:

Periodically retrain the model with new data to adapt to changing fraud patterns.

Enhance Feature Engineering:

Add more features (e.g., user behavior patterns, location data).

Deploy in Production:

Use a cloud platform (e.g., AWS, GCP) for scalable deployment.
