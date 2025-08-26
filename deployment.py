import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('deployment1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel_type = int(request.form['fuel_type'])

        # Calculate car age
        current_year = datetime.datetime.now().year
        car_age = current_year - year

        # Predict
        features = np.array([[car_age, km_driven, fuel_type]])
        prediction = model.predict(features)
        price = max(min(prediction[0], 10.0), 0)  # Clamp between ₹0 and ₹10 Lakhs

        return render_template('deployment1.html', prediction_text=f"Predicted Price: ₹{price:.2f} Lakhs")

    except Exception as e:
        return render_template('deployment1.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)



    

