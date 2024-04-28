Breast Cancer Detection Project

This repository contains code for a breast cancer detection project using machine learning techniques. The project aims to classify breast cancer tumors as malignant or benign based on features extracted from digitized images of breast tissue.

Features:

Data preprocessing: The dataset is loaded and preprocessed, including handling missing values and encoding categorical variables.
Exploratory Data Analysis (EDA): Various visualizations such as pairplots, countplots, and heatmaps are created to understand the data distribution and correlations.
Model Training: Several classification algorithms including Logistic Regression, Random Forest, AdaBoost, XGBoost, CatBoost, and Support Vector Machine (SVM) are trained on the dataset.
Hyperparameter Tuning: Grid search is performed to find the optimal hyperparameters for each model, enhancing their performance.
Model Evaluation: The trained models are evaluated using various metrics such as accuracy, precision, recall, and F1-score.
Receiver Operating Characteristic (ROC) Curve: ROC curves are plotted to compare the performance of different models in terms of true positive rate against false positive rate.
Usage:

Clone the repository: git clone https://github.com/amir-rs/Brest-Cancer-Prediction-using-ML.git
Run the Jupyter Notebook: jupyter notebook breast_cancer_detection.ipynb
Dataset:
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset available in scikit-learn's datasets module. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Contributing:
Contributions to improve the project are welcome. Feel free to open an issue or pull request with suggestions, bug fixes, or enhancements.
