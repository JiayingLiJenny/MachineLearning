from sklearn.tree import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
'''
最常见的超参数为：
base_estimator: 弱学习器使用的模型（切勿忘记导入该模型）。
n_estimators: 使用的弱学习器的最大数量。
'''
# 我们定义了一个模型，它使用 max_depth 为 2 的决策树作为弱学习器，并且它允许的弱学习器的最大数量为 4。
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators=4)
model.fit(x_train, y_train)
model.predict(x_test)
