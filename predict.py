import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data for predictive maintenance
np.random.seed(0)

# Sample size
num_samples = 100

# Sensor data: Vibration, Temperature, Pressure
vibration = np.random.uniform(5.8, 10, num_samples)
temperature = np.random.uniform(30, 100, num_samples)
pressure = np.random.uniform(8.2, 12, num_samples)

# Failure: 0 (No Failure), 1 (Failure)
failure = np.random.choice([0, 1], size=num_samples)

# Combine into features (X) and label (y)
X = np.column_stack((vibration, temperature, pressure))
y = failure

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=47)
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Predictive Maintenance Model Accuracy:", accuracy)

# Predict for a new sample
new_data = np.array([[9.5, 75, 10.2]])  # Vibration, Temperature, Pressure
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Prediction: Failure expected.")
else:
    print("Prediction: No Failure expected.")