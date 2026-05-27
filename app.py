from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn

# Create FastAPI app
app = FastAPI()

# Load trained model
model = pickle.load(open("disease_model.pkl", "rb"))

# Home Route
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API Running"
    }

# Prediction Route
@app.post("/predict")
def predict(area: float, bedrooms: int, age: int):

    # Convert data into array
    data = np.array([[area, bedrooms, age]])

    # Predict
    prediction = model.predict(data)

    # Return output
    return {
        "Predicted_Price": float(prediction[0])
    }

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)