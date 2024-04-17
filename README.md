# Bank Marketing Prediction

## Overview
This repository contains the implementation of various predictive models aimed at predicting subscription to term deposits by clients of a bank. The models analyze client data from direct marketing campaigns (phone calls) to predict whether clients will subscribe to term deposits.

## Problem Statement
The goal is to develop a predictive model that can accurately predict whether a client will subscribe ('yes') or not ('no') to a term deposit as part of direct marketing campaigns conducted by a Portuguese banking institution.

## Dataset
The dataset involves records from direct marketing campaigns of a bank, which are primarily phone calls. There are 45,210 samples with 17 features, including job type, marital status, education level, and previous campaign outcomes.

### Features
- **Categorical**: Job, marital status, education, default, housing, loan, contact, month, poutcome.
- **Numerical**: Age, balance, day, duration, campaign, pdays, previous.

### Target Variable
- **Deposit**: Whether the client subscribed to a term deposit ('yes' or 'no').

## Models Implemented
### Linear Models
- **Logistic Regression**
  - Accuracy: 0.885838
  - Kappa: 0.2219557
- **Linear Discriminant Analysis (LDA)**
  - Accuracy: 0.883468
  - Kappa: 0.3063123
- **Partial Least Squares Discriminant Analysis (PLSDA)**
  - Tuning Parameter (ncomp): 1
  - Accuracy: 0.8829942
  - Kappa: 0.0
- **Penalized Models**
  - Tuning Parameters: alpha = 0.1, lambda = 0.01
  - Accuracy: 0.8861540
  - Kappa: 0.1599455310

### Non-linear Models
- **Nonlinear Discriminant Analysis (NDLA)**
  - Tuning Parameter (subclasses): 5
  - Accuracy: 0.8764211
  - Kappa: 0.3153305
- **Neural Networks**
  - Tuning Parameters: size = 9, decay = 0
  - Accuracy: 0.8896926
  - Kappa: 0.29577028
- **Flexible Discriminant Analysis (FDA)**
  - Tuning Parameters: degree = 3, nprune = 15
  - Accuracy: 0.8815093
  - Kappa: 0.3637292
- **Support Vector Machines (SVM)**
  - Tuning Parameters: sigma = 0.05006803, C = 1
  - Accuracy: 0.8871020
  - Kappa: 0.1921608
- **k-Nearest Neighbors (KNN)**
  - Tuning Parameter (k): 3
  - Accuracy: 0.8696602
  - Kappa: 0.2586158
- **Naïve Bayes**
  - Accuracy: 0.8734198
  - Kappa: 0.357957

Results
Models are evaluated using accuracy and Kappa statistics. The Flexible Discriminant Analysis (FDA) model achieved the highest Kappa among all models, indicating superior performance on the imbalanced dataset.

Contributing
Contributions are welcome! Please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License - see the `LICENSE` file for details.
