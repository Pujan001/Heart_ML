Heart Disease Predictiont

This project analyzes a heart disease dataset and builds multiple machine-learning models to predict whether a patient is likely to have heart disease. The work is implemented in a Jupyter Notebook (ps.ipynb) using Python, Pandas, and Scikit-Learn.

Project Structure
ps.ipynb                 # Main Jupyter Notebook
heart_Dataset.csv        # Dataset

1️. Dataset Loading
Loads heart_Dataset.csv using pandas.
Displays:
First 5 rows
Dataset shape
Column names
Statistical summary (describe())

2️. Exploratory Data Analysis (EDA)
Inspects missing values
Reviews distributions
Understands dataset structure before training models

3️. Data Preprocessing
Separates features (X) and target (y)
Splits data into train/test sets using train_test_split()

4️. Machine Learning Models Used
The notebook trains and evaluates several classifiers:
 Gaussian Naive Bayes
 Logistic Regression
 K-Nearest Neighbors (KNN)
 Random Forest Classifier
 
5️. Model Evaluation
Each model outputs:
Accuracy score
Classification report (precision, recall, F1-score)

Technologies Used:
Tool / Library	Purpose
Python	Main programming language
Pandas	Data loading & preprocessing
NumPy	Numerical operations
Scikit-Learn	Machine learning models & evaluation
Jupyter Notebook	Environment for analysis & visualization

Results:
Each classifier produces its own performance metrics.
You can scroll through the notebook to compare which algorithm gives the highest accuracy and most balanced classification report.

How to Run the Notebook:
Install required libraries:
pip install pandas numpy scikit-learn jupyter
Open Jupyter Notebook:
jupyter notebook
Run all cells inside ps.ipynb.

Future Improvements:
Add visualizations (heatmaps, pairplots, ROC curves)
Add hyperparameter tuning (GridSearchCV)
Include feature importance analysis
Deploy model using Flask/Streamlit
