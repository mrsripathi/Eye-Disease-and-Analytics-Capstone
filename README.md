💡 Eye Disease and Analytics Capstone Project

This repository showcases four comprehensive machine learning projects built as part of the M19M20 & M21M22 Final Capstone by Sripathi D. The aim is to solve real-world problems using supervised and unsupervised learning with Python, PyTorch, and scikit-learn.

📁 Project Directory

1. 🧠 Eye Disease Classification

A deep learning project that detects eye diseases from retinal images using CNN.

Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal

Dataset: ~1000 images per class from datasets like IDRiD, HRF

Tools: PyTorch, TorchVision, CNN, BatchNorm, Dropout

Metrics: Confusion Matrix, Precision, Recall, F1-score

Script: Eye deseases.py trains the model and saves it as eye_cnn_model.pth

2. 🏦 Loan Status Prediction

A binary classification task predicting whether a loan will be approved.

Features: ApplicantIncome, Education, Dependents, Credit History, etc.

Preprocessing: Imputation, Encoding, Scaling, Feature Engineering

Models: Logistic Regression, Decision Tree, Random Forest, XGBoost

Evaluation: Accuracy comparison of different models

Script: Loan Status Prediction.py

3. 📊 Sales Forecasting

Regression models to forecast weekly retail department sales.

Data: Historical sales, store metadata, macroeconomic indicators, markdowns

Preprocessing: Log transformation, feature engineering, normalization

Models: Random Forest, PyTorch DNN

Metrics: MAE, RMSE

Scripts: Sale Forecasting model.py, exalple_code.py

4. 💳 Credit Card Clustering

Customer segmentation using unsupervised learning on credit card usage patterns.

Features: Balance, Purchase Frequency, Credit Limit, Tenure, etc.

Techniques: KMeans, PCA, Silhouette Score

Goal: Identify customer segments for marketing insights

Output: Cluster profiles, PCA visualizations, Elbow method plot

Script: credit card. clustering.py