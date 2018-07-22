from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_values, y_values)

print(model.predict([127],[248])) #predict() 是模型的函数
#用 [127] 这样的数组（而不只是 127）进行预测的原因是模型可以使用多个特征进行预测。我们将在这节课的稍后部分讲解如何在线性回归中使用多个变量。暂时先继续使用一个值。

####################################################
#练习：使用线性回归根据体质指数 (BMI) 预测预期寿命

# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']]) 
#注意不是bmi_life_data['BMI'],虽然这也能得到相同结果； 用两个【】表示里面可以接收多列，即多个特征，里面通常是series

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)# 这里为predict([21.07931])也是对的