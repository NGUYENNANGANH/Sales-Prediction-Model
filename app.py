
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load pre-trained models
linear_model = joblib.load('savefile/linear_regression_model.pkl')
ridge_model = joblib.load('savefile/ridge_regression_model.pkl')
mlp_model = joblib.load('savefile/mlp_regression_model1.pkl')
stacking_model = joblib.load('savefile/stacking_regressor_model.pkl')  
# Load scaler for X
scaler_X = joblib.load('savefile/scaler_X.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    model_type = data.get('model_type')
    # Map input form fields to feature names used in training
    features = {
        'TV_Ad_Budget_($)': float(data.get('feature1')),
        'Radio_Ad_Budget_($)': float(data.get('feature2')),
        'Newspaper_Ad_Budget_($)': float(data.get('feature3')),
    }
    # Convert features into a DataFrame and scale
    input_data = pd.DataFrame([features])
    input_data_std = pd.DataFrame(scaler_X.transform(input_data), columns=input_data.columns)
    # Generate predictions based on model type
    if model_type == 'linear':
        prediction = linear_model.predict(input_data_std)
    elif model_type == 'ridge':
        prediction = ridge_model.predict(input_data_std)
    elif model_type == 'mlp':
        prediction = mlp_model.predict(input_data_std)
    elif model_type == 'stacking':
        # Generate meta features for stacking model
        meta_features = np.column_stack((
            linear_model.predict(input_data_std),
            mlp_model.predict(input_data_std),
            ridge_model.predict(input_data_std)
        ))
        # Predict using the stacking model
        prediction = stacking_model.predict(meta_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    # Return prediction without scaling back since the target was not normalized
    prediction_original = prediction[0]
    return jsonify({'prediction': prediction_original})

if __name__ == '__main__':
    app.run(debug=True)




