#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a Flask web application
app = Flask(__name__, template_folder='templates')

# Load your dataset
heart_df = pd.read_csv("framingham.csv")
heart_df.drop(['education'], axis=1, inplace=True)
heart_df.dropna(axis=0, inplace=True)

# Define the machine learning model (Logistic Regression)
new_features = heart_df[['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = new_features.iloc[:, :-1]
y = new_features.iloc[:, -1]
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x, y)

# Define a route to serve an HTML form for user input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
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

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

