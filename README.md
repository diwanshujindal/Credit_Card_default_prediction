# Credit Card Default Prediction

## Project Overview
This project aims to predict whether a credit card client will default on their payment in the next month. The analysis includes exploratory data analysis (EDA), data preprocessing, and building machine learning models to predict default behavior.

## Dataset
The dataset used is "Default of Credit Card Clients" from the UCI Machine Learning Repository. It contains information on 30,000 credit card clients with 24 features including demographic information, payment history, bill statements, and previous payment amounts.

## Features
- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1 = male, 2 = female)
- **EDUCATION**: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others)
- **MARRIAGE**: Marital status (1 = married, 2 = single, 3 = others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Repayment status for previous 6 months
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts for previous 6 months
- **PAY_AMT1 to PAY_AMT6**: Payment amounts for previous 6 months
- **default payment next month**: Target variable (1 = default, 0 = non-default)

## Project Structure

credit-card-default-prediction/
├── data/ # Dataset files

│   ├── default of credit card clients.xls # Original dataset.

├── notebooks/ # Jupyter notebooks

│   └── Credit_card_default_prediction.ipynb # Main analysis notebook

├── src

│   ├── __init__.py # Package initialization

│   ├── data_preprocessing.py # Data loading and preprocessing functions

│   ├── eda.py # Exploratory data analysis functions

│   └── modeling.py # Model training and evaluation functions

├── requirements.txt # Python dependencies

└── README.md # Project documentation
