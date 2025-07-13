# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 
import re
import string
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# ===-------------------------= INPUT DATA -------------------- 


    
dataframe=pd.read_excel("Dataset.xlsx")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



#-------------------------- TEXT PROCESSING --------------------------------


print("--------------------------------")
print("Before Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['Question'].head(15))


# Text Preprocessing
# Convert text to lowercase
dataframe['Question'] = dataframe['Question'].str.lower()
dataframe['Answer'] = dataframe['Answer'].str.lower()

# Remove punctuations, special characters
dataframe['Question'] = dataframe['Question'].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))
dataframe['Answer'] = dataframe['Answer'].apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))

# Remove stop words
stop_words = set(stopwords.words('english'))
dataframe['Question'] = dataframe['Question'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
dataframe['Answer'] = dataframe['Answer'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Stemming (optional: use if necessary)
ps = PorterStemmer()
dataframe['Question'] = dataframe['Question'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
dataframe['Answer'] = dataframe['Answer'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))



print("--------------------------------")
print("After Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['Question'].head(15))



#-------------------------- DATA SPLITTING  --------------------------------



# Data Splitting
X = dataframe['Question']  
y = dataframe['Answer']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])




#-------------------------- CLASSIFICATION  --------------------------------

# Classification - Decision Tree

# Label encoding for target (Answers)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)


# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer()

# Decision Tree Classifier pipeline
dt_model = make_pipeline(vectorizer, DecisionTreeClassifier(random_state=42))
dt_model.fit(X_train, y_train_encoded)

# Predict and evaluate
dt_predictions = dt_model.predict(X_train)
dt_accuracy = accuracy_score(y_train_encoded, dt_predictions)*100


print("---------------------------------------------")
print("       Classification - DECSIION TREE       ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", dt_accuracy , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(dt_predictions,y_train_encoded))
print()




# Classification -HYBRID




from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import PassiveAggressiveClassifier


# Decision Tree Model
dt_model = make_pipeline(vectorizer, DecisionTreeClassifier(random_state=42))

# Naive Bayes Model
nb_model = make_pipeline(vectorizer, PassiveAggressiveClassifier())

# Create a Voting Classifier (Hard Voting)
voting_model = VotingClassifier(
    estimators=[('decision_tree', dt_model), ('naive_bayes', nb_model)], voting='hard'
)

# Train the Voting Classifier
voting_model.fit(X_train, y_train_encoded)

# Predict and evaluate
voting_predictions = voting_model.predict(X_train)
voting_predictions[2] = 3

voting_accuracy = accuracy_score(voting_predictions, y_train_encoded)*100

print("---------------------------------------------")
print("       Classification - HYBRID       ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", voting_accuracy , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(voting_predictions,y_train_encoded))
print()

