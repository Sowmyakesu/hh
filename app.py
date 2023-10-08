#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import pandas as pd
import openai
from flask import Flask, render_template, request, jsonify  # Import necessary Flask modules
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
# Set your API key as an environment variable
import os
os.environ["OPENAI_API_KEY"] = "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"

# Define a route to handle API requests
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    # Get user input from the form
    user_input = request.form['user_input']

    # Debug: Print user input
    print("User Input:", user_input)

    # Make an API request to ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides suggestions."},
            {"role": "user", "content": user_input}
        ]
    )

    # Debug: Print API response
    print("API Response:", response)

    # Extract the assistant's reply
    suggestion = response['choices'][0]['message']['content']

    # Return the suggestion as JSON
    return jsonify({'suggestion': suggestion})


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
    app.run(debug=True, port=8080)


