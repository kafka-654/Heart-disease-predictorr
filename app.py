import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def main():
    st.title("Heart Disease Classification Web App")
    st.sidebar.title("Heart Disease Classification Web App")
    st.markdown("Do you have a heart disease? ")
    st.sidebar.markdown("Do you have a Heart Disease?")
    
    def load_data():
        data = pd.read_csv("heart1.csv")
        return data
    
    def split(df):
        y = df.target
        x = df.drop(columns = ["target"])
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            #fig, ax = plt.subplots()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()
        
        if "ROC curve" in metrics_list:
            st.subheader("ROC Curve")
            #fig, ax = plt.subplots()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precion recall curve")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
        
    
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["Disease","No Disease"]
    
    if st.sidebar.checkbox("Show data set", False):
        st.subheader("Heart disease Data set")
        st.write(df)
    
    
    if st.sidebar.checkbox("Apply different models on the dataset", False):
        st.subheader("Choose Classifier")
        classifier = st.selectbox("Classifier", ("Support Vector Machine(SVM)","Logistic Regression", "Random Forests" ))
        
        if classifier == "Support Vector Machine(SVM)":
            st.subheader("Modsel Hyperparameters")
            C = st.number_input("C (regularization parameter)", 0.01, 10.00, step = 0.01, key = "C")
            kernel = st.radio("Kernel",("rbf", "linear"), key = "kernel")
            gamma = st.radio("Gamma (kernel coefficient)", ("scale", "auto"), key = "gamma")
            
            metrics = st.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC curve", "Precision-Recall Curve"))
            
            if st.button("Classify", key= "classify"):
                st.subheader("SVM results :")
                model = SVC(C=C, kernel = kernel, gamma = gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names ).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names ).round(2))
                plot_metrics(metrics)
                
        if classifier == "Logistic Regression":
            st.subheader("Model Hyperparameters")
            C = st.number_input("C (regularization parameter)", 0.01, 10.00, step = 0.01, key = "C_lr")
            max_iter = st.slider("Maximum number of iterations :", 100, 500, key = "max_iter")
            
            
            metrics = st.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC curve", "Precision-Recall Curve"))
            
            if st.button("Classify", key= "classify"):
                st.subheader("Logistic Regression results :")
                model = LogisticRegression(C=C,max_iter = max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names ).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names ).round(2))
                plot_metrics(metrics)
                
        if classifier == "Random Forests":
            st.subheader("Model Hyperparameters")        
            n_estimators = st.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
            max_depth = st.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
            bootstrap = st.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
            metrics = st.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC curve', 'Precision-Recall Curve'))
        
            if st.button("Classify", key = 'classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
                plot_metrics(metrics)
            
    
    
        
    def load_data1():
        data = pd.read_csv("heart1.csv")
        return data
    
    if st.sidebar.checkbox("Make your own prediction (Using random forests)", False):
        st.subheader("Enter the features")
        age = st.number_input("age:", value=63, min_value=0, max_value=100, step = 1, key = 'age')
        sex = st.number_input("sex:", value=1, min_value=0, max_value=1, step = 1, key = 'sex')
        cp = st.number_input("cp:", value=3, min_value=0, max_value=3, step = 1, key = 'cp')
        trest_bps = st.number_input("trest_bps:", value=145, min_value=100, max_value=200, step = 1, key = 'trest_bps')
        chol = st.number_input("chol:", value=233, min_value=100, max_value=450, step = 1, key = 'chol')
        fbs = st.number_input("fbs:", value=1, min_value=0, max_value=1, step = 1, key = 'fbs')
        rest_ecg = st.number_input("rest_ecg:", value=0, min_value=0, max_value=1, step = 1, key = 'rest_ecg')
        thalach = st.number_input("thalach:", value=150, min_value=80, max_value=210, step = 1, key = 'thalach')
        exang = st.number_input("exang:", value=0, min_value=0, max_value=1, step = 1, key = 'exang')
        old_peak = st.number_input("old_peak:", value=2.30, min_value=0.0, max_value=7.0, step = 0.1, key = 'old_peak')
        slope = st.number_input("slope:", value=0, min_value=0, max_value=2, step = 1, key = 'slope')
        ca = st.number_input("ca:", value=0, min_value=0, max_value=4, step = 1, key = 'ca')
        thal = st.number_input("thal:", value=1, min_value=0, max_value=3, step = 1, key = 'thal')
        if st.button("Predict", False):
           df1 = load_data1()
           x_train, x_test, y_train, y_test = split(df1)
        
           model = RandomForestClassifier()
           model.fit(x_train, y_train)
           arr = np.array([age,sex,cp,trest_bps,chol,fbs,rest_ecg,thalach,exang,old_peak,slope,ca,thal])
           prediction = (model.predict_proba([arr])[0][1])*100
           st.write("The probability of having a heart disease is ", prediction,"%." )   
            
    
    
        
    































if __name__ == '__main__':
    main()
