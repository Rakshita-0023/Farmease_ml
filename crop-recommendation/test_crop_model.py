import joblib

# Load trained model
model = joblib.load("crop_model.pkl")

# Example input (soil + weather values)
sample_input = [[
    90,    # Nitrogen
    42,    # Phosphorus
    43,    # Potassium
    20.8,  # temperature
    82.0,  # humidity
    6.5,   # pH
    202    # rainfall
]]

# Predict crop
prediction = model.predict(sample_input)

print("🌾 Recommended Crop:", prediction[0])
