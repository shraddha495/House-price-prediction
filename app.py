import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the model relative to the current script directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Built-in modern template with glassmorphism, shadow effects, and a categorical dropdown
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-dark: #0f172a;
            --text-muted: #64748b;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 12px;
            box-sizing: border-box;
        }

        .card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 32px;
            width: 100%;
            max-width: 480px;
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.4),
                0 8px 10px -6px rgba(0, 0, 0, 0.3),
                0 0 0 1px rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
        }

        .title {
            margin: 0 0 8px 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-dark);
            text-align: center;
        }

        .subtitle {
            margin: 0 0 24px 0;
            font-size: 0.875rem;
            color: var(--text-muted);
            text-align: center;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .full-width {
            grid-column: span 2;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            padding: 10px 12px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 0.95rem;
            color: var(--text-dark);
            background-color: #f8fafc;
            transition: all 0.2s ease;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.03);
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #ffffff;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }

        button {
            margin-top: 12px;
            padding: 14px;
            border: none;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        button:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 20px;
            padding: 16px;
            border-radius: 10px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 700;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
    </style>
</head>
<body>
    <div class="card">
        <h1 class="title">Property Predictor</h1>
        <p class="subtitle">Enter specs below to get an instant valuation</p>
        
        <form action="/" method="POST">
            <div class="form-grid">
                <div class="form-group full-width">
                    <label for="square_footage">Square Footage</label>
                    <input type="number" step="any" id="square_footage" name="Square_Footage" placeholder="e.g. 2100" required>
                </div>

                <div class="form-group">
                    <label for="num_bedrooms">Bedrooms</label>
                    <input type="number" step="any" id="num_bedrooms" name="Num_Bedrooms" placeholder="3" required>
                </div>

                <div class="form-group">
                    <label for="num_bathrooms">Bathrooms</label>
                    <input type="number" step="any" id="num_bathrooms" name="Num_Bathrooms" placeholder="2" required>
                </div>

                <div class="form-group">
                    <label for="year_built">Year Built</label>
                    <input type="number" step="any" id="year_built" name="Year_Built" placeholder="2015" required>
                </div>

                <div class="form-group">
                    <label for="garage_size">Garage (Cars)</label>
                    <input type="number" step="any" id="garage_size" name="Garage_Size" placeholder="2" required>
                </div>

                <div class="form-group full-width">
                    <label for="lot_size">Lot Size (sq ft)</label>
                    <input type="number" step="any" id="lot_size" name="Lot_Size" placeholder="e.g. 5000" required>
                </div>

                <div class="form-group full-width">
                    <label for="neighborhood_quality">Neighborhood Category</label>
                    <select id="neighborhood_quality" name="Neighborhood_Quality" required>
                        <option value="" disabled selected>Select Quality</option>
                        <option value="1">Low</option>
                        <option value="2">Medium</option>
                        <option value="3">High</option>
                        <option value="4">Premium</option>
                    </select>
                </div>

                <button type="submit" class="full-width">Calculate Value</button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="result-box">
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
        # Array ordered exactly to match the feature vector expected by linear.pkl
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        # Predict price
        prediction = model.predict([features])[0]
        prediction_text = f"Estimated Value: ${prediction:,.2f}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

# Required handler for Vercel WSGI deployment
app = app.wsgi_app

if __name__ == '__main__':
    app.run(debug=True)
