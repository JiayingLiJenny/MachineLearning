def doPCA():
	from sklearndecomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca

pca = doPCA()

# the explained variance ratio is actually where the eigenvalues live
print pca.explained_variance_ratio_ 
#方差比，是特征值的具体表现形式
# 输出：[0.90774318, 0.09225682]
# 通过输出可以得到，第一个主成分占数据变动的90%，而第二个主成分占比约9%
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_data = pca.transform(data)

# 进行可视化
for ii, jj in zip(transformed_data, data):
	plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color="r") #第一主成分
	plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], color="c") #第二主成分
	plt.scatter(jj[0], jj[1], color="b") # 可视化数据点
plt.xlabel("bonus")
plt.ylabel("log-term incentive")
plt.show()




























