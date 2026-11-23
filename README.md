# smart-fitness-recommendation-system
Machine learning project for smart fitness and meal recommendation.

This is a machine learning project developed. The project aims to build a smart fitness and meal recommendation system by analysing user fitness data and generating personalised recommendations.

## My Contribution

My main contribution was the data science and machine learning component of the project. I worked on data cleaning, dataset integration, missing value handling, BMI consistency checks, feature engineering, model testing, clustering, and connecting fitness profiles with meal recommendation logic.

## Methods Used

- Data cleaning and preprocessing
- Feature engineering
- Regression
- k-Nearest Neighbours (kNN)
- Random Forest
- K-Means clustering
- Rule-based recommendation logic
- Model evaluation using R² and RMSE

## Engineered Features

- Session duration
- BMI
- BPM increase
- MET

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Outcome

The project analysed fitness user profiles and connected machine learning-based fitness profiling with a rule-based meal recommendation system.

## Model Evaluation

The notebook compares four regression models for estimating calories
burned during a workout.

| Model | R² | RMSE (kcal) |
| --- | ---: | ---: |
| Random Forest Regression | 0.18 | 273.74 |
| kNN Regression | 0.17 | 275.36 |
| Multiple Linear Regression | 0.12 | 283.40 |
| Degree-2 Polynomial Regression | 0.08 | 290.79 |

Random forest recorded the lowest RMSE in this comparison, narrowly
ahead of kNN. With an R² of 0.18, the predictions are best interpreted
as rough estimates.

## Note

This repository is a portfolio version of a group academic project. Some materials may have been simplified or reorganised for presentation purposes.
