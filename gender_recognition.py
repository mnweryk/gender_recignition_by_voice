import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def program(size):
    data = pandas.read_csv("voice.csv")
    #print(data)
    y = data.label.values                                   #zmienna objaśniana
    x_data = data.drop(["label"], axis=1)                   #zmienna objaśniająca
    #x = x_data.values
    x = (x_data - np.mean(x_data))/(np.std(x_data)).values  #Standaryzacja
    #print(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
    #print(x_train.shape[0])
    prediction = LogisticRegression().fit(x_train, y_train).predict(x_test)
    if(size==0.2):
        print(prediction-y_test)                          #1 - przewidział fałszywie kobietę

    return size, accuracy_score(prediction, y_test)



results = []

for i in range(1,100, 1):
    if(10%i==0 or i%2==0):
        results.append(program(i/100))

for result in results:
    print(result)