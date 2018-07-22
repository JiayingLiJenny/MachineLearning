from sklearn.svm import SVC #C-Support Vector Classification
'''
超参数:

C：C 参数。
kernel：内核。最常见的内核为 'linear'、'poly' 和 'rbf'。
degree：如果内核是多项式，则此参数为内核中的最大单项式次数。
gamma：如果内核是径向基函数，则此参数为 γ 参数。
'''
model = SVC(kernel='poly', degree=4, C=0.1)
model.fit(x_values, y_values)

print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
>>> [[ 0., 1.]] #每个输入数组一个预测结果

#####################################################
练习

# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=27)

# TODO: Fit the model.
model.fit(X,y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)




