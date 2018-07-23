from sklearn.preprocessing import MinMaxScaler
import numpy
weights = numpy.aray([[115.],[140.],[175.]]) # 必须为浮点数，整数后面要加个点
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
# fit_transform()这里进行了两步：
# fit：找出最大最小值进行公式计算
# transform：把数组中所有值进行缩放
