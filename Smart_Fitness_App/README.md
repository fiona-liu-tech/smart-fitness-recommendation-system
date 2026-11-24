# Smart Fitness & Meal Planner

An interactive web app based on the Smart Fitness & Meal Recommendation
course project. Enter a workout and optional meal preferences. The app
estimates calories burned, identifies a fitness profile and suggests up
to five meals with nutrition information.

## Run locally on Windows

Use Python 3.12 or 3.13. Open this folder in VS Code and open a PowerShell
terminal in the same folder as `app.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser. The app fits the models on
startup, so wait for the server startup message. Click **Get recommendations**
with the pre-filled example, then try a different workout or meal preference.
Press Ctrl+C in the terminal to stop the server.

## Models and recommendations

- Random forest uses the original notebook’s feature set and seeded 70/30
  split. This app's test run returned R² 0.182 and RMSE 273.74 kcal.
  Treat the calorie predictions as rough estimates.
- K-means uses two clusters on scaled workout features available before a
  prediction. The actual calorie-burn field is excluded from clustering.
- Meal selection reuses the notebook's burn levels and meal categories.
  Recommendations are ranked by proximity to the estimate. Explicit diet,
  meal type and cuisine filters apply throughout.
- The `data/` folder contains the three CSVs from the supplied project.
  Suggestions are for demonstration and do not account for food allergies.

## Files

- `app.py` — FastAPI routes and input validation
- `model.py` — data preparation, training and meal selection
- `web/index.html` — browser interface
