
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
linear_model = joblib.load('savefile/linear_regression_model.pkl')
ridge_model = joblib.load('savefile/ridge_regression_model.pkl')
mlp_model = joblib.load('savefile/mlp_regression_model1.pkl')
stacking_model = joblib.load('savefile/stacking_model.pkl')

# Load scaler for X only
scaler_X = joblib.load('savefile/scaler_X.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    model_type = data.get('model_type')

    # Map the input form fields to the correct feature names used in training
    features = {
        'TV_Ad_Budget_($)': float(data.get('feature1')),
        'Radio_Ad_Budget_($)': float(data.get('feature2')),
        'Newspaper_Ad_Budget_($)': float(data.get('feature3')),
    }

    # Convert the features into a DataFrame
    input_data = pd.DataFrame([features])

    # Scale the input data and ensure it retains the feature names
    input_data_std = pd.DataFrame(scaler_X.transform(input_data), columns=input_data.columns)

    # Make prediction using the selected model
    if model_type == 'linear':
        prediction = linear_model.predict(input_data_std)
    elif model_type == 'ridge':
        prediction = ridge_model.predict(input_data_std)
    elif model_type == 'mlp':
        prediction = mlp_model.predict(input_data_std)
    elif model_type == 'stacking':
        # Remove feature names for stacking model
        # input_data_std_np = input_data_std.to_numpy()
        prediction = stacking_model.predict(input_data_std)
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    # Dự đoán không cần chuyển đổi lại thang đo vì không chuẩn hóa Y
    prediction_original = prediction[0]  # Dự đoán đã ở thang đo gốc

    return jsonify({'prediction': prediction_original})

if __name__ == '__main__':
    app.run(debug=True)
