#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


heart_df = pd.read_csv(r"C:\Users\komma\Desktop\framingham.csv")


# In[3]:


heart_df.drop(['education'], axis=1, inplace=True)
heart_df.head()
# Assuming you have already dropped the 'education' column
#print("Number of Rows:", heart_df.shape[0])  # Number of rows
#print("Number of Columns:", heart_df.shape[1])  # Number of columns


# In[4]:


# Check for missing values in each column
missing_values = heart_df.isnull().sum()
#print(missing_values)
heart_df.dropna(axis=0,inplace=True)
# Check for missing values in each column
missing_values = heart_df.isnull().sum()
#print(missing_values)


# In[5]:


from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(heart_df)
#heart_df_constant.head()


# In[6]:


import statsmodels.api as sm
from scipy import stats

# Define the chisqprob function
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# Continue with the rest of your code
cols = heart_df_constant.columns[:-1]
model = sm.Logit(heart_df.TenYearCHD, heart_df_constant[cols])
result = model.fit()
#result.summary()


# In[15]:


import statsmodels.api as sm

def backward_feature_elimination(data_frame, dep_var, col_list):
    results = []

    while len(col_list) > 0:
        model = sm.Logit(dep_var, data_frame[col_list])
        result = model.fit(disp=0)
        largest_pvalue = round(result.pvalues, 3).nlargest(1)
        
        if largest_pvalue[0] < 0.05:
            results.append(result)
            col_list = col_list.drop(largest_pvalue.index)
        else:
            break
    
    return results

results = backward_feature_elimination(heart_df_constant, heart_df.TenYearCHD, cols)


# In[17]:


import numpy as np
params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
#print ((conf))


# In[18]:


from sklearn.model_selection import train_test_split
import sklearn
new_features = heart_df[['age', 'male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'TenYearCHD']]
x = new_features.iloc[:, :-1]
y = new_features.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)


# In[19]:


from sklearn.linear_model import LogisticRegression
#logreg=LogisticRegression()
# Specify an alternative solver
logreg = LogisticRegression(solver='liblinear')
# Initialize logistic regression with feature names

# Fit the model with training data
logreg.fit(x_train, y_train)


logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# In[20]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
#print("Accuracy:", accuracy)
#print("Confusion Matrix:\n", conf_matrix)
#print("Classification Report:\n", class_report)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sn

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

#plt.figure(figsize=(8, 5))
#sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
#plt.show()


# In[22]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
#print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

#'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

#'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',
#'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

#'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

#'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

#'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

#'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)
y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()


# In[23]:


# # Define a dictionary to map user input feature names to their descriptions
feature_descriptions = {
    'age': 'Age (in years): ',
    'sex': 'Gender (0 for female, 1 for male): ',
    'cigsPerDay': 'Number of cigarettes per day: ',
    'totChol': 'Total Cholesterol (mg/dL): ',
    'sysBP': 'Systolic Blood Pressure (mm Hg): ',
    'glucose': 'Glucose Level (mg/dL): '
}

# Accept user inputs for each feature
user_inputs = {}
for feature, description in feature_descriptions.items():
    user_input = input(description)
    # Convert the user input to the appropriate data type (float for numeric features)
    user_inputs[feature] = float(user_input)

# Create a NumPy array from the user inputs
user_data = np.array([list(user_inputs.values())])

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

# Display results to the user
print("\nHeart Disease Risk Assessment:")
print(f"Risk Level: {risk_level}")
print(f"Heart Disease Probability: {heart_disease_prob[0]:.2f}")
print("\nSuggestions:")
for i, suggestion in enumerate(suggestions, start=1):
    print(f"{i}. {suggestion}")


# In[ ]:




