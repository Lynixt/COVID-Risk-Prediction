# importing Flask and other modules 
from flask import Flask, request, render_template 
import numpy as np
import pandas as pd

# list of attributes
l1=['O-blood-group','child-adult','medical-complication','international-travel','contact-with-covid','covid-symptoms']

# classes
disease=['Very-high','Pretty-High','High','Pretty-low','low']

# TESTING DATA df 
df=pd.read_csv("dataset/test.csv")

df.replace({'prognosis':{'Very-high':0,'Pretty-High':1,'High':2,'Pretty-low':3,'low':4}},inplace=True)

# print(df.head())

X= df[l1]
y = df[["prognosis"]]
np.ravel(y)


# TRAINING DATA tr 
tr=pd.read_csv("dataset/train.csv")
tr.replace({'prognosis':{'Very-high':0,'Pretty-High':1,'High':2,'Pretty-low':3,'low':4}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# --------------------

# Flask constructor 
app = Flask(__name__) 

#Decision Tree Algorithm
def DecisionTree(psymptoms):
    #import tree 
    from sklearn import tree

    # empty model of the decision tree
    clf3 = tree.DecisionTreeClassifier()  
    clf3 = clf3.fit(X,y)

    # calculating accuracy-----------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    # --------------------------------------------

    # Prediction
    inputtest = [psymptoms]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    # check if prediction is done successfully 
    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes' 
            break
        
    if (h=='yes'):
        return(disease[a], accuracy_score(y_test, y_pred,normalize=False)) 
    else:
        return("Not Found", "Not Found")

#Naive Bayes Algorithm
def NaiveBayes(psymptoms):
    # import gaussian
    from sklearn.naive_bayes import GaussianNB

    # empty model of naive gaussian
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy--------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    # ----------------------------------------------

    # Prediction
    inputtest = [psymptoms]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    # check if prediction is done successfully 
    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'

            break

    if (h=='yes'):
        return(disease[a], accuracy_score(y_test, y_pred,normalize=False)) 
    else:
        return("Not Found", "Not Found")


@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Take user inputs from html form
        name = request.form.get("name")
        age = int(request.form.get("age"))
        blood_group = int(request.form.get("blood_group"))
        medical_complication = int(request.form.get("medical_complication"))
        travel_history = int(request.form.get("travel_history"))
        covid_contact = int(request.form.get("covid_contact"))
        covid_symptoms = int(request.form.get("covid_symptoms"))
        psymptoms = [age, blood_group, medical_complication, travel_history, covid_contact, covid_symptoms]
        naive_prediction , nb_accuracy = NaiveBayes(psymptoms)
        d_t_prediction, dt_accuracy = DecisionTree(psymptoms)
        
    return render_template("index.html", name = name, prediction_nb = naive_prediction,accuracy_nb = nb_accuracy , prediction_dt = d_t_prediction, accuracy_dt = dt_accuracy)
	
 
if __name__=='__main__':  
    app.run(debug=True) 
