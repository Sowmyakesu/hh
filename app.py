#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a Flask web application
app = Flask(__name__, template_folder='templates')


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load your dataset with a relative file path
dataset_path = os.path.join(current_dir, 'framingham.csv')
heart_df = pd.read_csv(dataset_path)
heart_df.drop(['education'], axis=1, inplace=True)
heart_df.dropna(axis=0, inplace=True)

# Define the machine learning model (Logistic Regression)
new_features = heart_df[['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = new_features.iloc[:, :-1]
y = new_features.iloc[:, -1]
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x, y)


# Define a route to serve an HTML form for user input using GET
@app.route('/', methods=['GET'])
def get_index():
    return render_template('index.html')

import requests

def get_suggestions(input_text):
    api_key = '- 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'
    endpoint = 'https://api.openai.com/v1/engines/davinci-codex/completions'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'prompt': input_text,
        'max_tokens': 50,  # Adjust based on your needs
    }

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        suggestions = result['choices'][0]['text']
        return suggestions
    else:
        # Handle API error
        return 'Error fetching suggestions'

# Usage example in your Flask route
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions_route():
    user_input = request.form['user_input']
    suggestions = get_suggestions(user_input)
    return suggestions

# Define a route to handle form submission using POST
@app.route('/', methods=['POST'])
def post_index():
    # Get user input from the form (similar to your existing POST route)
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cigsPerDay = float(request.form['cigsPerDay'])
    totChol = float(request.form['totChol'])
    sysBP = float(request.form['sysBP'])
    glucose = float(request.form['glucose'])

    # Create a NumPy array from user inputs
    user_data = np.array([[age, sex, cigsPerDay, totChol, sysBP, glucose]])

    # Use the trained model to predict heart disease risk
    heart_disease_prob = logreg.predict_proba(user_data)[:, 1]

    # Determine risk level and provide suggestions
    if heart_disease_prob < 0.2:
        risk_level = "Low"
        suggestions = ["Maintain a healthy lifestyle.", "Regularly monitor your health."]
    elif 0.2 <= heart_disease_prob < 0.6:
        risk_level = "Moderate"
        suggestions = ["Consult with a healthcare professional for personalized advice.",
                       "Consider making lifestyle changes like diet and exercise."]
    else:
        risk_level = "High"
        suggestions = ["Urgently consult with a healthcare professional.", "Follow medical advice and treatment."]

    return render_template('result.html', risk_level=risk_level, heart_disease_prob=heart_disease_prob[0], suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)

