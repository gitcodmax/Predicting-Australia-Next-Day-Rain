# Predict Next Day Rain 

## 📌 Project Overview 

This project focuses on using historical weather data collected from different locations across Australia
to predict whether it will rain on the following day. The project applies machine learning classification 
techniques to identify patterns in weather conditions such as temperature, humidity, rainfall, wind speed, 
cloud cover, and atmospheric pressure that may be associated with next-day rainfall.

## 🎯 Problem Statement

Weather conditions change across locations and over time, making it difficult to accurately determine whether 
rainfall will occur on the following day. The objective of this project is to develop a machine learning model 
that uses historical daily weather observations to predict whether it will rain the next day, classifying the 
outcome as either Yes or No.

## 📊 Dataset

The project uses the Weather in Australia (weatherAUS) dataset, which contains daily weather observations 
collected from numerous Australian weather stations. It contains approximately 145460 observations across 23 columns.

The target variable is ```RainTomorrow``` which contains one of two values, Yes or No. 

Some of the main features in the dataset are: 

| Feature | Description |
| --- | --- |
| ```Date``` | Date of the weather observation |
| ```Location``` | Weather station location |
| ```MinTemp``` | Minimum temperature |
| ```Rainfall``` | Rainfall recorded during the day |
| ```WindDir9am``` | Wind direction at 9am |
| ```Cloud3pm``` | Cloud cover at 3pm |
| ```RainToday``` | Whether it rained on the current day | 

## 🔎 Exploratory Data Analysis  

The insights gathered are:
- ```RainToday``` and ```RainTomorrow```, which we expect to hold a lot of weight on whether it will rain the next
 day contain a lot of null values, 4673
- Most cities did not experience rain on the day of data collection
- Class imbalance: In most of the days there was no rain recorded
- On days that it did not rain, high temperatures were recorded
- Dataset contains records from 2008 to 2017 with the highest records on 2016

## 🛠️ Data Preprocessing 

The following activities were carried to prepare the data for modelling:
- Drop the null records in the ```RainToday``` and ```RainTomorrow``` features to deal with the real values from
  the two important columns. Also by dropping the records we are still left with enough records to learn the
  patterns and make strong predictions.
- Data leakage: In ```notebooks/log_reg_est.ipynb```, all the records were used during the preprocessing stages instead of
  using only the training test. In ```notebooks/log_reg_dt_rf_xgb_est.ipynb```, only the training set was used during
  preprocessing ensuring there is NO DATA LEAKAGE.
- Handling null values: Fill null values with average in ```notebooks/log_reg_est.ipynb``` and 
  median in ```notebooks/log_reg_dt_rf_xgb_est.ipynb```
- Scaling numerical values: Numerical values are scaled using MinMaxScaler-```notebooks/log_reg_est.ipynb``` and
  StandardScaler(z-score)-```notebooks/log_reg_dt_rf_xgb_est.ipynb```
- Categorical values are encoded through one hot encoding
- Splitting data using ```Date``` for model evaluation: training-<2015, validation-2015 and test->2015 sets

## 🤖 Models

Different algorithms were used to train models and this is how they performed.  
Scoring metric: Accuracy score

| Model | Training score | Validation score | Test Score |
| --- | --- | --- | --- |
| Base model | 77% | 79% | 77% |
| Logistic regression | 85% | 85% | 84% | 
| Decision trees | 85% | 84% | 83% |
| Random forest | 86% | 85% | 84% |
| XGBoost | 89% | 86% | 85% |

Best model: XGBOOST 
Model Evaluation: The models built were very good at predicting the days that it did not rain compared to when it rained.
This was confirmed by the values from the confusion matrix.

## 📁 File Descriptions

| File/Folder | Description |
| --- | --- |
| ```data/``` | All the datasets used in the project |
| ```data/processed/``` | Preprocessed data ready for model training |
| ```data/raw/``` | Raw datasets used |
| ```model/``` | Trained model |
| ```notebooks/``` | Notebooks and python files used for modelling |

```                               MMAX CODES                                                                   ```
