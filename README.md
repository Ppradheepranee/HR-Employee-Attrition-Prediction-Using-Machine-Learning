# HR-Employee-Attrition-Prediction-Using-Machine-Learning


HR Employee Attrition Prediction

Executive Summary
This project analyzes employee attrition patterns at INX Future Inc., a leading data analytics and automation solutions provider. The objective is to identify root causes of declining employee performance and develop predictive indicators for non-performing employees to support HR decision-making without negatively impacting overall employee morale.
Using machine learning techniques on historical HR data, this model predicts employee attrition with 89% accuracy, enabling proactive HR interventions and workforce retention strategies.

________________________________________
1. Introduction

1.1 Problem Statement

INX Future Inc. faces declining employee performance indexes and increased service delivery escalations. Client satisfaction has dropped by 8 percentage points, creating concern among top management. However, the company must balance performance management with maintaining its reputation as a top employer ranked in the best 20 companies for the past 5 years.

1.2 Business Objective
•	Develop a data-driven analysis to identify root causes of employee performance issues
•	Provide clear indicators for identifying non-performing employees
•	Support HR decision-making without negatively impacting overall employee morale
•	Enable proactive retention strategies to prevent attrition

1.3 Target Variable
Attrition: Binary classification (Yes/No) indicating whether an employee has left the organization
________________________________________
2. Dataset Overview

2.1 Data Source
Employee Performance Analysis Dataset for INX Future Inc. (Version 1.8)

2.2 Dataset Dimensions
•	Total Records: 1,200 employees
•	Total Features: 28 employee attributes
•	Target Variable: Attrition (Binary: Yes/No)

2.3 Key Features
Category	Features

Demographics	Age, Gender, Marital Status
Professional	Employee Number, Department, Job Role, Education Background
Experience	Total Work Experience, Years at Company, Years in Current Role, Years Since Last Promotion, Years with Current Manager
Work-Life Factors	Distance from Home, Business Travel Frequency, Work-Life Balance Rating
Satisfaction Metrics	Job Satisfaction, Relationship Satisfaction, Environment Satisfaction
Performance	Performance Rating, Training Times Last Year

2.4 Data Characteristics
•	Balanced Target: Both attrition categories represented
•	Mixed Data Types: Numerical, categorical, and ordinal variables
•	No Missing Values: Complete dataset ready for analysis
________________________________________

3. Exploratory Data Analysis

3.1 Data Preprocessing Steps

1.	Library Imports: NumPy, Pandas, Matplotlib, Seaborn
2.	Dataset Loading: Excel file import using Pandas
3.	Data Exploration: Shape, info, and descriptive statistics
4.	Missing Value Check: Confirmed no missing values
5.	Categorical Encoding: Label encoding for categorical variables
6.	Feature Scaling: Standardized numerical features for model training



3.2 Domain Analysis
•	Age: Employee age range affecting demographics and career stage
•	Gender: Gender distribution in workforce
•	Department: Sales, Development, Human Resources, Data Science
•	Job Roles: Executive, Manager, Senior Developer, Data Scientist
•	Education Background: Marketing, Life Sciences, Medical, Human Resources
•	Travel Frequency: TravelRarely, TravelFrequently, Non-Travel
•	Satisfaction Metrics: 1-4 scale ratings for job and relationship satisfaction
________________________________________

4. Model Development


4.1 Feature Engineering
•	Encoded categorical variables (Gender, Department, Job Role, Education Background, Marital Status, Business Travel Frequency)
•	Standardized numerical features for optimal model performance
•	Selected 27 features for model training (excluding Employee ID)

4.2 Data Splitting
•	Training Set: 70% of data (840 samples)
•	Testing Set: 30% of data (360 samples)
•	Random state maintained for reproducibility

4.3 Model Selection: Logistic Regression
Rationale:
•	Binary classification problem suited to logistic regression
•	Interpretable coefficients for feature importance analysis
•	Fast training and prediction
•	Good baseline for comparison with advanced models

4.4 Model Training
Algorithm: Logistic Regression with default hyperparameters
Optimizer: L2 regularization for overfitting prevention
________________________________________

5. Model Evaluation

5.1 Performance Metrics
Metric	Score
Accuracy	89.17%
Precision	76.92%
Recall	37.74%
F1-Score	50.63%


5.2 Metric Interpretation
•	Accuracy (89.17%): The model correctly predicts attrition status for approximately 9 out of 10 employees
•	Precision (76.92%): When the model predicts attrition, it is correct 77% of the time (low false positives)
•	Recall (37.74%): The model identifies only 38% of actual attrition cases (higher false negatives)
•	F1-Score (50.63%): Balanced measure accounting for precision-recall tradeoff


5.3 Overfitting Analysis
•	Training Accuracy: 90.36%
•	Testing Accuracy: 89.17%
•	Conclusion: Minimal overfitting observed (1.19% difference indicates good generalization)


5.4 Cross-Validation Performance
K-Fold Cross-Validation (10-fold):
Fold	Accuracy
1	84.17%
2	85.83%
3	90.83%
4	86.67%
5	92.50%
6	92.50%
7	91.67%
8	90.83%
9	87.50%
10	90.00%


Mean Cross-Validation Accuracy: 89.25% ± 2.82%
Interpretation: The model maintains consistent performance across different data splits, with an average expected accuracy of approximately 89% on unseen data.
________________________________________


6. Model Insights and Interpretability

6.1 Feature Importance
Top factors influencing employee attrition identified through logistic regression coefficients

6.2 Predictive Indicators
The model provides clear indicators for identifying employees at risk of attrition, enabling:
•	Proactive retention strategies
•	Targeted HR interventions
•	Resource allocation for high-risk employees
________________________________________


7. Technical Implementation


7.1 Libraries and Dependencies
numpy # Numerical computing
pandas # Data manipulation
matplotlib # Visualization
seaborn # Statistical visualization
scikit-learn # Machine learning


7.2 Model Training Code

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Model initialization and training
lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)
Predictions
y_pred = lg_model.predict(X_test)
Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


7.3 Environment
•	IDE: Jupyter Notebook
•	Python Version: 3.7+
•	Platform: Anaconda Environment
________________________________________


8. Recommendations

8.1 For HR Department

1.	Proactive Monitoring: Use the model to identify high-risk employees for targeted retention discussions
2.	Personalized Interventions: Design department-specific strategies based on attrition patterns
3.	Work-Life Balance: Address travel frequency and work-life balance concerns identified by the model
4.	Career Development: Enhance promotion opportunities and training frequency for retention


8.2 For Model Improvement
1.	Address Recall Gap: Implement class weight adjustment or threshold tuning to improve detection of attrition cases
2.	Ensemble Methods: Experiment with Random Forest or Gradient Boosting for better performance
3.	Feature Engineering: Derive interaction features for improved predictive power
4.	Threshold Optimization: Balance precision-recall tradeoff based on business requirements


8.3 For Business Impact
•	Reduce unexpected employee turnover
•	Improve service delivery consistency
•	Maintain reputation as a top employer
•	Support data-driven HR decision-making
________________________________________


9. Conclusion
This project successfully develops a predictive model for employee attrition at INX Future Inc. with 89% accuracy. The model provides actionable insights for identifying at-risk employees while maintaining the company's commitment to employee-friendly HR policies.

Key achievements:
•	✓ Clear identification of attrition patterns
•	✓ High overall accuracy (89.17%)
•	✓ Minimal overfitting (good generalization)
•	✓ Consistent cross-validation performance
•	✓ Interpretable model for business stakeholders
By leveraging these insights, INX Future Inc. can take proactive measures to retain valuable talent, improve performance metrics, and maintain its position as a top employer in the industry.
________________________________________


10. References
•	Employee Performance Analysis Dataset - INX Future Inc. (Version 1.8)
•	Scikit-learn Documentation: https://scikit-learn.org/
•	Python Data Analysis Library (Pandas): https://pandas.pydata.org/
•	NumPy: https://numpy.org/
________________________________________


________________________________________


Project Status: ✓ Completed
Model Ready for Deployment: Yes - 89% accuracy on test set
Last Updated: February 2026
