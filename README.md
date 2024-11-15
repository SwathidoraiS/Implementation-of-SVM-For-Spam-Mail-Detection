# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Swathi. S
RegisterNumber:  212223040220
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Encoding:
![image](https://github.com/user-attachments/assets/ffc731f6-1920-4e33-861c-24b211c88437)

## Head():

![image](https://github.com/user-attachments/assets/ed58f233-53e0-4eb9-9c1e-9c509ee84287)

## Info():
![image](https://github.com/user-attachments/assets/d057a84a-390c-4c6a-a65b-a311e95676f1)

## isnull().sum():

![image](https://github.com/user-attachments/assets/a714f497-08b3-49c2-b4f3-51cadf9e4b55)

## Prediction of y:

![image](https://github.com/user-attachments/assets/bf8c537e-599b-4c18-bd13-c8cc58e8fa53)

## Accuracy:

![image](https://github.com/user-attachments/assets/31cb78f7-4d5c-4bf9-9332-62a2e922b778)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
