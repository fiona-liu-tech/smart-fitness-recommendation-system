# Smart Fitness & Meal Recommendation System

A machine learning portfolio project with an interactive web app that
estimates calories burned during a workout and suggests meals based on
the estimate, fitness profile, and dietary preferences.

[App source and local setup instructions](Smart_Fitness_App/README.md)

## What the App Does

- Estimates workout calorie expenditure from inputs such as workout type,
  duration, heart rate, and self-reported training experience.
- Uses a two-cluster K-means model to assign a fitness profile.
- Suggests up to five meals and displays their calories, protein,
  carbohydrates, and fat. Users can filter by diet, meal type, and cuisine.

The app uses FastAPI for the backend and HTML, CSS, and JavaScript for
the interface. The Jupyter notebook documents the data preparation,
exploratory analysis, and model comparison.

## My Contribution

This portfolio project extends a group coursework project. My main
contribution was the data science and machine learning work. I cleaned
and combined two workout datasets, checked BMI consistency, engineered
features such as heart-rate increase and session duration in minutes,
compared regression models, and linked fitness profiles to rule-based
meal recommendations.

## Model Evaluation

| Regression model | Test R² | Test RMSE (kcal) |
| --- | ---: | ---: |
| Random forest | 0.18 | 273.74 |
| k-nearest neighbours | 0.17 | 275.36 |
| Multiple linear regression | 0.12 | 283.40 |
| Degree-2 polynomial regression | 0.08 | 290.79 |

Random forest had the lowest test RMSE and was selected for the app.
Its test R² of 0.18 indicates limited predictive performance, so the app
presents calorie predictions as rough estimates.

## Run Locally

Open the `Smart_Fitness_App` folder and follow the
[setup instructions](Smart_Fitness_App/README.md). Once the server starts,
open `http://127.0.0.1:8000` and click **Get recommendations** to try the
pre-filled example.
