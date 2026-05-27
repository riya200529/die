import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("cicd_logistic/disease_data.csv")

# Features and Target
x = df[['age', 'bp', 'sugar']]
y = df['disease']

# Split Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model
model = LogisticRegression()

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# Save Model
pickle.dump(model, open("disease_model.pkl", "wb"))

# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model Saved")