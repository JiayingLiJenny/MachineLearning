### 1. 导入 GridSearchCV
from sklearn.model_selection import GridSearchCV
### 2.选择参数：
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
### 3.创建一个评分机制 (scorer)
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)
### 4. 使用参数 (parameter) 和评分机制 (scorer) 创建一个 GridSearch 对象。 使用此对象与数据保持一致 （fit the data) 。
# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)
### 5. 获得最佳估算器 (estimator)
best_clf = grid_fit.best_estimator_