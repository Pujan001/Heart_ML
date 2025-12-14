 Heart Disease Prediction Using Machine Learning

 Project Description:
This project aims to predict the presence of **heart disease** using multiple **machine learning classification models**. The implementation is done in a Jupyter Notebook (`ps.ipynb`) using Python and popular machine learning libraries.  
Several models are trained, evaluated, and compared to determine which performs best on the given dataset.

 Objectives:
- Load and preprocess the heart disease dataset
- Split data into training and testing sets
- Train multiple machine learning models
- Evaluate model performance using standard metrics
- Compare results to identify the most accurate model

 Machine Learning Models Implemented:
The following models are implemented in the notebook:

- **LogR_model** – Logistic Regression  
- **LR_model** – Linear Regression  
- **RF_model** – Random Forest Classifier  
- **ET_model** – Extra Trees Classifier  
- **GB_model** – Gradient Boosting Classifier  
- **SVM_model** – Support Vector Machine  
- **XG_model** – XGBoost Classifier  
Additionally, **GridSearchCV** is used for **hyperparameter tuning** (especially for Random Forest).

 Dataset:
- The dataset contains patient medical attributes such as:
  - Age
  - Sex
  - Blood pressure
  - Cholesterol
  - Heart rate
  - Other clinical features
- Target variable:
  - `1` → Presence of heart disease  
  - `0` → No heart disease

 Methodology:
1. Import required libraries  
2. Load the dataset using Pandas  
3. Perform data preprocessing and feature selection  
4. Split data into training and testing sets (80% train, 20% test)  
5. Train multiple ML models  
6. Evaluate models using performance metrics  
7. Compare results across models  

 Evaluation Metrics:
Each model is evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report:
  - Precision
  - Recall
  - F1-score

 Results & Observations:
- Ensemble models such as Random Forest, Extra Trees, Gradient Boosting and XGBoost showed strong predictive performance.
- Logistic Regression served as a reliable baseline model.
- Hyperparameter tuning improved Random Forest performance.
- Model comparison helped identify the most effective classifier for heart disease prediction.
  
 Tools & Technologies
- Language: Python  
- Environment: Jupyter Notebook  
- Libraries Used:
  - NumPy
  - Pandas
  - Scikit-learn
  - XGBoost
  - Matplotlib
