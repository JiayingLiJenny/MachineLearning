from sklearn import datasets, cluster
# load dataset
X = datasets.load_iris().data[:10]

clust = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')
# 'ward' linkage is default, can also use 'complete' or 'average'.

labels = clust.fit_predict(X)
# 'labels' contains an array representing which cluster each point belongs to:
# [1 0 0 0 1 2 0 1 0 0]



# 绘制系统树
from scipy.cluster.hierarchy import dendrogram, ward, single
from sklearn import datasets
import matplotlib.pyplot as plt
 X = datasets.load_iris().data[:10]
# perform clustering
linkage_matrix = ward(X)
# plot dendogram
dendogram(linkage_matrix)

plt.show()


# DBSCAN
from sklearn import datasets, cluster
X = datasets.load_iris().data

db cluster.DBSCAN(eps=0.5, min_samples=5)
db.fit(X)
# 'labels' contains an array representing which cluster each point belongs to
# labeled '-1' are noise


# GaussianMixture
from sklearn import datasets, mixture
X = datasets.load_iris().data[:10]

gmm = mixture.GaussianMixture(n_components=3)
gmm.fix(X)
clustering = gmm.predict(X)

# 也可以写成
# from sklearn.mixture import GaussianMixture
# gmm = GaussianMixture(n_components=3, random_state=0)




















