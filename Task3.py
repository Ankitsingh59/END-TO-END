# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import requests
import joblib
import os
from pydantic import BaseModel
from typing import List

# ================================================================
# 1. DATA COLLECTION & PREPROCESSING
# ================================================================

# Define the URL for the Mushroom dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
DATASET_FILE = "agaricus-lepiota.csv"

def get_data():
    """Fetches the dataset from the URL if it doesn't exist locally."""
    if not os.path.exists(DATASET_FILE):
        print("Downloading dataset...")
        try:
            response = requests.get(DATASET_URL)
            with open(DATASET_FILE, 'wb') as f:
                f.write(response.content)
            print("Dataset downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    # Load the dataset into a pandas DataFrame
    # The dataset has no header, so we provide column names
    column_names = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    df = pd.read_csv(DATASET_FILE, header=None, names=column_names)
    return df

df = get_data()

if df is None:
    raise RuntimeError("Failed to load dataset.")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# The 'veil-type' column has only one unique value ('p'), so it's not useful
# for classification and will cause an error in one-hot encoding
X = X.drop('veil-type', axis=1)

# All columns are categorical. We use a preprocessor to handle this.
# A pipeline is used to chain together the one-hot encoding and the model training
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), X.columns)
    ])

# ================================================================
# 2. MODEL TRAINING
# ================================================================

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with the preprocessor and the classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model (for demonstration purposes)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model accuracy on test data: {accuracy:.4f}")

# Save the trained model and the feature names
joblib.dump(model_pipeline, 'mushroom_model.joblib')
joblib.dump(list(X.columns), 'features.joblib')
print("Model and feature names saved.")

# ================================================================
# 3. MODEL DEPLOYMENT WITH FASTAPI
# ================================================================

# Load the trained model and feature names
model = joblib.load('mushroom_model.joblib')
feature_names = joblib.load('features.joblib')
categorical_values = {col: list(X[col].unique()) for col in feature_names}

app = FastAPI()

# Mount the static directory for CSS and JS files
app.mount("/static", StaticFiles(directory="."), name="static")

# Setup Jinja2Templates to render HTML files
templates = Jinja2Templates(directory=".")

# Define a Pydantic model for the prediction request
# This ensures that the incoming data has the correct structure and types
class MushroomFeatures(BaseModel):
    features: List[str]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the main HTML page for the application.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categorical_values": categorical_values
        }
    )

@app.post("/predict")
async def predict_mushroom(features: MushroomFeatures):
    """
    Prediction endpoint that takes a list of categorical features and
    returns whether the mushroom is edible or poisonous.
    """
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([features.features], columns=feature_names)
    
    try:
        # Get the prediction from the model pipeline
        prediction = model.predict(input_data)[0]
        
        # 'p' for poisonous, 'e' for edible
        result = "poisonous" if prediction == 'p' else "edible"
        
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# ================================================================
# 4. USER INTERFACE (HTML)
# ================================================================

# This HTML content is provided as a separate file for clarity.
# When running the app, it should be in a file named "index.html"
# in the same directory.
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mushroom Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-2xl max-w-xl w-full">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Mushroom Classifier</h1>
        <p class="text-gray-600 text-center mb-8">
            Select the characteristics of a mushroom to predict if it is edible or poisonous.
        </p>

        <form id="prediction-form" class="space-y-4">
            {% for feature, values in categorical_values.items() %}
            <div class="flex flex-col">
                <label for="{{ feature }}" class="text-sm font-medium text-gray-700 mb-1 capitalize">{{ feature.replace('-', ' ') }}:</label>
                <select id="{{ feature }}" name="{{ feature }}" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md shadow-sm">
                    {% for value in values %}
                    <option value="{{ value }}">{{ value }}</option>
                    {% endfor %}
                </select>
            </div>
            {% endfor %}

            <div class="flex justify-center pt-6">
                <button type="submit" class="w-full sm:w-auto px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition duration-150 ease-in-out">
                    Predict Edibility
                </button>
            </div>
        </form>

        <div id="prediction-result" class="mt-8 text-center hidden">
            <h2 class="text-2xl font-semibold text-gray-800">Prediction:</h2>
            <p id="result-text" class="mt-2 text-3xl font-bold"></p>
        </div>
        <div id="error-message" class="mt-4 text-center text-red-500 hidden"></div>
    </div>

    <script>
        const form = document.getElementById('prediction-form');
        const resultDiv = document.getElementById('prediction-result');
        const resultText = document.getElementById('result-text');
        const errorDiv = document.getElementById('error-message');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            
            resultDiv.classList.add('hidden');
            errorDiv.classList.add('hidden');
            resultText.textContent = '';
            
            const formData = new FormData(form);
            const features = Array.from(formData.values());

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features: features }),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Something went wrong.');
                }

                const data = await response.json();
                
                resultText.textContent = data.prediction.toUpperCase();
                resultDiv.classList.remove('hidden');

                if (data.prediction === 'poisonous') {
                    resultText.classList.remove('text-green-600');
                    resultText.classList.add('text-red-600');
                } else {
                    resultText.classList.remove('text-red-600');
                    resultText.classList.add('text-green-600');
                }

            } catch (error) {
                errorDiv.textContent = `Error: ${error.message}`;
                errorDiv.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
"""

# Save the HTML content to a file for use with Jinja2
with open("index.html", "w") as f:
    f.write(HTML_CONTENT)

# To run this application:
# 1. Save this code as `app.py` in a new directory.
# 2. Make sure `index.html` is created in the same directory from the HTML_CONTENT above.
# 3. Create a virtual environment and install the required libraries:
#    pip install fastapi pandas scikit-learn uvicorn requests python-multipart jinja2
# 4. Run the application from your terminal:
#    uvicorn app:app --reload
# 5. Open your web browser and navigate to http://127.0.0.1:8000