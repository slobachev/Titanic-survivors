import numpy as np
import pandas as pd
from sklearn import tree
import os

location = r'D:\Projects\Titanic\titanic.csv'
df = pd.read_csv(location)

df = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']][~df.Age.isnull()]
selection = df[['Pclass', 'Fare', 'Age', 'Sex']]

selection['Sex'] = [0 if item == 'male' else 1 for item in selection['Sex']]

survived = df['Survived']

clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(selection, survived)
print(clf.feature_importances_)
