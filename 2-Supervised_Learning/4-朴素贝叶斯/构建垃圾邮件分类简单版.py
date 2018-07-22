此项目分为以下步骤：

第 0 步: 朴素贝叶斯定理简介
第 1.1 步: 了解我们的数据集
第 1.2 步: 数据预处理
第 2.1 步: Bag of Words(BoW)
第 2.2 步: 从头实现 BoW
第 2.3 步: 在 scikit-learn 中实现 Bag of Words
第 3.1 步: 训练和测试数据集
第 3.2 步: 向我们的数据集中应用 Bag of Words 处理流程
第 4.1 步: 从头实现贝叶斯定理
第 4.2 步: 从头实现朴素贝叶斯定理
第 5 步: 使用 scikit-learn 实现朴素贝叶斯定理
第 6 步: 评估模型
第 7 步: 结论

import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()


mapping = {'ham':0, 'spam':1}
df['label'] = df.label.map(mapping)
#df['label'] = df['label'].map(mapping)
#df['label'] = df.label.map({'ham':0, 'spam':1})
#以上三种写法都是对的，但运行第二次会出现NaN情况，因为运行第一遍已经是1，0了，再运行一次映射不到导致出现NaN。
print('num of rows are {}, num of columns are {}.'.format(*df.shape))
df.head()

#############################################################
#先以文档集合为例
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']



from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)

doc_names = count_vector.get_feature_names()
doc_array = count_vector.transform(documents).toarray()

frequency_matrix = pd.DataFrame(doc_array, columns=doc_names)


######################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))


'''
The code for this segment is in 2 parts. Firstly, we are learning a vocabulary dictionary for the training data 
and then transforming the data into a document-term matrix; secondly, for the testing data we are only 
transforming the data into a document-term matrix.
'''
from sklearn.feature_extraction.text import CountVectorizer
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
# 需要return matrix，所以使用fit_transform而不是fit
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

'''
Instructions:
Now that our algorithm has been trained using the training data set we can now make some predictions on the test data
stored in 'testing_data' using predict(). 
'''
predictions = naive_bayes.predict(testing_data)

#评估模型
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))













