import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"D:\2025\flask\flask_react\data\diabetes.csv")
# print(df)
# print(df.columns)

X = df.drop('Outcome', axis = 1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 2)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

# knn_score = knn.score(X_test,y_test)

pickle.dump(knn, open('diabetes_model_knn',"wb"))

print("Training and Saving completed")