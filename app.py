import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Built-in single-file HTML layout with custom modern styles and shadow effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --text-main: #1e293b;
            --text-muted: #64748b;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 20px;
        }

        .container {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 35px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 
                        0 8px 10px -6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        h2 {
            margin-top: 0;
            color: var(--text-main);
            font-size: 1.75rem;
            text-align: center;
            margin-bottom: 24px;
        }

        .form-group {
            margin-bottom: 16px;
        }

        label {
            display: block;
            margin-bottom: 6px;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.875rem;
        }

        input, select {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
        }

        button {
            width: 100%;
            padding: 12px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: background-color 0.2s ease, transform 0.1s ease;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4);
        }

        button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        .result {
            margin-top: 24px;
            padding: 16px;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            color: #166534;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>House Price Predictor</h2>
        <form action="/" method="POST">
            <div class="form-group">
                <label for="square_footage">Square Footage</label>
                <input type="number" step="any" id="square_footage" name="Square_Footage" required>
            </div>
            <div class="form-group">
                <label for="num_bedrooms">Num Bedrooms</label>
                <input type="number" step="any" id="num_bedrooms" name="Num_Bedrooms" required>
            </div>
            <div class="form-group">
                <label for="num_bathrooms">Num Bathrooms</label>
                <input type="number" step="any" id="num_bathrooms" name="Num_Bathrooms" required>
            </div>
            <div class="form-group">
                <label for="year_built">Year Built</label>
                <input type="number" step="any" id="year_built" name="Year_Built" required>
            </div>
            <div class="form-group">
                <label for="lot_size">Lot Size</label>
                <input type="number" step="any" id="lot_size" name="Lot_Size" required>
            </div>
            <div class="form-group">
                <label for="garage_size">Garage Size</label>
                <input type="number" step="any" id="garage_size" name="Garage_Size" required>
            </div>
            <div class="form-group">
                <label for="neighborhood_quality">Neighborhood Quality</label>
                <select id="neighborhood_quality" name="Neighborhood_Quality" required>
                    <option value="" disabled selected>Select Quality</option>
                    <option value="1">Low</option>
                    <option value="2">Medium</option>
                    <option value="3">High</option>
                </select>
            </div>
            <button type="submit">Predict Price</button>
        </form>
        
        {% if prediction_text %}
        <div class="result">
            {{ prediction_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        # Feature inputs matching your model's exact metadata order:
        # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        prediction = model.predict([features])[0]
        prediction_text = f"Estimated Price: ${prediction:,.2f}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
