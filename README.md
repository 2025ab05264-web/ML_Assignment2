Problem Statement:  Binary classification to predict if an individual's annual income exceeds $50k based on census data.

b. Dataset Description: Source: UCI Adult Income Dataset (via Google Drive).Features: 14 features including age, education, and occupation (Minimum requirement: 12).Instances: Over 30,000 records after cleaning (Minimum requirement: 500)

.c. Models Used Comparison Table: 
Use the table printed by the code above. This section is worth 6 marks.+1d. Observations on Performance: Fill this table for the final 3 marks:+1ML Model NameObservation about model performance Logistic RegressionLinear baseline; fast but may struggle with complex non-linear relationships.Decision TreeHigh interpretability; prone to overfitting on high-cardinality features.kNNInstance-based learning; sensitive to feature scaling and computationally heavy for large sets.Naive BayesHigh speed; assumes feature independence which may not hold for census data.Random ForestRobust bagging ensemble; reduces variance and handles outliers well.XGBoostPowerful boosting ensemble; usually yields the highest AUC and MCC on this dataset.
