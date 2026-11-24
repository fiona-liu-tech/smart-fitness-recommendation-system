"""Training and meal preparation adapted from the project notebook."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent / "data"
TARGET = "Calories_Burned"
MET = {"Yoga": 3.0, "Strength": 5.0, "Cardio": 7.0, "HIIT": 8.5}
FEATURES = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "BMI",
    "Max_BPM",
    "Avg_BPM",
    "Session_Duration_Minutes",
    "Fat_Percentage",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BPM_Increase",
    "MET",
    "Gender_Female",
    "Gender_Male",
    "Workout_Type_Cardio",
    "Workout_Type_HIIT",
    "Workout_Type_Strength",
    "Workout_Type_Yoga"
]
CLUSTER_FEATURES = [
    "Age",
    "Weight (kg)",
    "BMI",
    "Avg_BPM",
    "Session_Duration_Minutes",
    "Calories_Burned",
    "Workout_Frequency (days/week)",
    "Experience_Level",
    "BPM_Increase",
    "MET"
]

def load_workouts():
    gym = pd.read_csv(DATA / "gym_members_exercise_tracking.csv")
    tracker = pd.read_csv(DATA / "Fitness Tracker Dataset.csv")
    for frame in (gym, tracker):
        for column in frame.columns.difference(["Gender", "Workout_Type"]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ("Gender", "Workout_Type"):
            frame[column] = frame[column].str.strip()
    bmi = (gym["Weight (kg)"] / gym["Height (m)"] ** 2).round(2)
    gym = gym.loc[bmi.eq(gym["BMI"].round(2))]
    tracker = tracker.loc[tracker["Workout_Type"].isin(MET)].copy()
    tracker["BMI"] = (tracker["Weight (kg)"] / tracker["Height (m)"] ** 2).round(2)
    workouts = pd.concat([gym, tracker.dropna()], ignore_index=True)
    workouts["Session_Duration_Minutes"] = (workouts["Session_Duration (hours)"] * 60).round(2)
    workouts["BPM_Increase"] = workouts["Avg_BPM"] - workouts["Resting_BPM"]
    workouts["MET"] = workouts["Workout_Type"].map(MET)
    return pd.get_dummies(workouts, columns=["Gender", "Workout_Type"], dtype=int)


def fit_models():
    workouts = load_workouts().dropna(subset=FEATURES + [TARGET])
    train = workouts.sample(frac=0.7, random_state=0)
    test = workouts.drop(train.index)
    forest = RandomForestRegressor(n_estimators=200, random_state=42)
    forest.fit(train[FEATURES], train[TARGET])
    predicted = forest.predict(test[FEATURES])
    scores = {
        "r2": round(r2_score(test[TARGET], predicted), 3),
        "rmse": round(mean_squared_error(test[TARGET], predicted) ** 0.5, 2),
    }
    # Use the same observable inputs for training and inference.
    profile_features = [column for column in CLUSTER_FEATURES if column != TARGET]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train[profile_features])
    clusters = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled)
    high_label = train.groupby(clusters.labels_)[["Avg_BPM", "BPM_Increase"]].mean().sum(axis=1).idxmax()
    return forest, scaler, clusters, profile_features, high_label, scores


def load_meals():
    meal_df = pd.read_csv(DATA / "healthy_eating_dataset.csv")
    text_cols = [
        "meal_name",
        "cuisine",
        "meal_type",
        "diet_type",
        "cooking_method"
    ]
    for col in text_cols:
        if col in meal_df.columns:
            meal_df[col] = meal_df[col].astype(str).str.strip()

    numeric_cols = [
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "cholesterol_mg",
        "serving_size_g"
    ]
    for col in numeric_cols:
        meal_df[col] = pd.to_numeric(meal_df[col], errors="coerce")
    meal_df = meal_df.dropna()
    meal_df = meal_df[meal_df["calories"] > 0]
    meal_df = meal_df[meal_df["is_healthy"] == 1]
    def meal_category(row):
        if row["calories"] < 350:
            return "Light Meal"
        elif row["calories"] < 650:
            return "Balanced Meal"
        else:
            return "High Energy Meal"
    meal_df["Meal_Category"] = meal_df.apply(meal_category, axis=1)
    meal_keep = [
        "meal_name",
        "cuisine",
        "meal_type",
        "diet_type",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "Meal_Category"
    ]
    meal_df = meal_df[meal_keep]
    return meal_df

def burn_level(calories):
    if calories < 700:
        return "Low Burn"
    elif calories < 1100:
        return "Medium Burn"
    else:
        return "High Burn"

recommendation_map = {
    "Low Burn": ["Light Meal"],
    "Medium Burn": ["Balanced Meal"],
    "High Burn": ["Balanced Meal", "High Energy Meal"]
}
profile_meal_map = {
    "High Intensity Fitness Profile": ["Balanced Meal", "High Energy Meal"],
    "Moderate Fitness Profile": ["Light Meal", "Balanced Meal"]
}

def apply_filters(df, diet_type="Autofill", meal_type="Autofill", cuisine="Autofill"):
    if diet_type != "Autofill":
        df = df[df["diet_type"] == diet_type]
    if meal_type != "Autofill":
        df = df[df["meal_type"] == meal_type]
    if cuisine != "Autofill":
        df = df[df["cuisine"] == cuisine]
    return df
def recommend(predicted_calories, profile, meals, filters):
    burn = burn_level(predicted_calories)
    categories = set(recommendation_map[burn]) | set(profile_meal_map[profile])
    matches = apply_filters(meals, **filters)
    matches = matches[matches["Meal_Category"].isin(categories)]
    within_range = matches["calories"].between(predicted_calories * 0.6, predicted_calories * 1.1)
    close_matches = matches[within_range]
    if len(close_matches) < 5:
        close_matches = matches[matches["calories"].between(predicted_calories * 0.4, predicted_calories * 1.4)]
    ranked = close_matches.assign(distance=(close_matches["calories"] - predicted_calories).abs())
    return burn, ranked.sort_values("distance", kind="stable").head(5).drop(columns="distance")
