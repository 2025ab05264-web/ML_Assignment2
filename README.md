**Problem Statement**
The Goal of this project is to build and deply a machine learning applicationBinary to predict weather an individual's annual income exceeds $50,000 based on census demographic data. This is binary classification problem.


**Dataset Description**
Source: UCI Adult Income Dataset 
Features: 14 features including age, education, and occupation (Minimum requirement: 12).Instances: Over 45,222 records after cleaning 
Target Variable: Income (<=50K or >50K)

Six classification Models were implemented:
1.  Logistic Regression
2.  Decesion Tree
3.  K-Nearest Neighbours (kNN)
4.  Naive Bayes
5.  Random Forest
6.  XGBoost


** COMPARISION TABLE **
|ML Model Name              | Accuracy | AUC| Precision | Recall |F1 | MCC |
|---------------------------|----------|---------|-----------|-------|--------|-------|
|Logistic Regression        |0.8211    |0.8485   |0.7115     |0.4467 |0.5488  |0.4632 |
|Decision Tree              |0.8064    |0.7420   |0.5998     |0.6164  |0.6080  |0.4796 |
|kNN                        |0.8276    |0.8545   |0.6655    |0.5878  |0.6242  |0.5146 |
|Naive Bayes                |0.7989    |0.8522   |0.6784    |0.3314  |0.4453  |0.3723 |
|Random Forest (Ensemble)   |0.8557    |0.9048   |0.7448    |0.6201  |0.6767  |0.5889 |
|XGBoost (Ensemble)        |0.8718  |0.9264      |0.7790    |0.6609  |0.7151  |0.6367 |

**Observation on Model Performance**

|ML Model Name              |Observations about Model Performance                                                                          |
|---------------------------|--------------------------------------------------------------------------------------------------------------|
|Logistic Regression        |Provides a solid baseline accuracy (82%), but has relatively low recall, missing many high-income earners.    |
|Decision Tree              |Captures non-linear patterns well but shows the lowest AUC (0.74), suggesting potential overfitting or instability.|
|kNN                        |Balanced performance across most metrics, benefiting from the normalized (scaled) demographic features.       |
|Naive Bayes                |Showed the lowest recall (0.33), indicating it struggles to correctly identify the positive class (>50K) in this dataset.|
|Random Forest              |Significantly outperformed individual trees, showing high stability with an AUC of 0.90.                      |
|XGBoost                    |Best Performer: Achieved the highest scores across all metrics (87% Accuracy, 0.926 AUC, 0.63 MCC).            |

