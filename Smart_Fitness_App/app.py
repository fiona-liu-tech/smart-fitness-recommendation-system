from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import FEATURES, MET, fit_models, load_meals, recommend

app = FastAPI(title="Smart Fitness & Meal Planner")
WEB = Path(__file__).resolve().parent / "web"
forest, scaler, clusters, profile_features, high_label, scores = fit_models()
meals = load_meals()
choices = {key: sorted(meals[key].unique().tolist()) for key in ("diet_type", "meal_type", "cuisine")}


class Workout(BaseModel):
    age: int = Field(ge=18, le=59)
    gender: str
    weight: float = Field(ge=40, le=130)
    height: float = Field(ge=1.5, le=2)
    workout: str
    duration: float = Field(ge=15, le=120)
    resting_bpm: int = Field(ge=40, le=100)
    avg_bpm: int = Field(ge=80, le=195)
    max_bpm: int = Field(ge=90, le=205)
    fat: float = Field(ge=8, le=45)
    frequency: int = Field(ge=1, le=7)
    experience: int = Field(ge=1, le=3)
    diet_type: str = "Autofill"
    meal_type: str = "Autofill"
    cuisine: str = "Autofill"


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/options")
def options():
    return {"filters": choices, "scores": scores}


@app.post("/api/recommend")
def prediction(workout: Workout):
    if workout.workout not in MET or workout.gender not in {"Female", "Male"}:
        raise HTTPException(422, "Choose a workout type and a supported model category.")
    if not workout.resting_bpm <= workout.avg_bpm <= workout.max_bpm:
        raise HTTPException(422, "Heart rates must be in order: resting ≤ average ≤ maximum.")
    filters = {key: getattr(workout, key) for key in choices}
    for key, value in filters.items():
        if value != "Autofill" and value not in choices[key]:
            raise HTTPException(422, "Choose one of the available meal filters.")
    row = {
        "Age": workout.age,
        "Weight (kg)": workout.weight,
        "Height (m)": workout.height,
        "BMI": round(workout.weight / workout.height ** 2, 2),
        "Max_BPM": workout.max_bpm,
        "Avg_BPM": workout.avg_bpm,
        "Session_Duration_Minutes": workout.duration,
        "Fat_Percentage": workout.fat,
        "Workout_Frequency (days/week)": workout.frequency,
        "Experience_Level": workout.experience,
        "BPM_Increase": workout.avg_bpm - workout.resting_bpm,
        "MET": MET[workout.workout],
    }
    row.update({f"Gender_{gender}": int(workout.gender == gender) for gender in ("Female", "Male")})
    row.update({f"Workout_Type_{kind}": int(workout.workout == kind) for kind in MET})
    inputs = pd.DataFrame([row], columns=FEATURES)
    calories = float(forest.predict(inputs)[0])
    label = clusters.predict(scaler.transform(inputs[profile_features]))[0]
    profile = "High Intensity Fitness Profile" if label == high_label else "Moderate Fitness Profile"
    burn, recommendations = recommend(calories, profile, meals, filters)
    return {
        "calories": round(calories),
        "profile": profile,
        "burn": burn,
        "meals": recommendations.to_dict(orient="records"),
    }
