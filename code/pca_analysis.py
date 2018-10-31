from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold

# from pandas.tools.plotting import parallel_coordinates, radviz
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def functionPCA(Xmatrix, n_clusters=3):
	seed = np.random.seed(0)
	colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
	colors = np.hstack([colors] * 20)

	# normalize dataset for easier parameter selection
	X = StandardScaler().fit_transform(Xmatrix)

	print(X)

	pca = PCA(n_components='mle')
	#kpca = KernelPCA(n_components=2, kernel='sigmoid')
	# X_kpca  = kpca.fit_transform(X)

	X_pca = pca.fit_transform(X)

	varExpl = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
	print("Variance Explained: ", str(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)))
	params = pca.get_params(deep=True)

	# select number of features to use on clustering (enough to exlpain 90% of the variance)
	if (varExpl[-1] > 95):
		n_features = np.where(varExpl > 95)[0][0]
	else:
		n_features = len(varExpl) - 1

	#kmeans
	# algorithm KMeans
	kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

	# Apply algorithm
	kmeans.fit(X_pca[:, 0:n_features + 1])

	y_pred = kmeans.labels_.astype(np.int)
	centers = kmeans.cluster_centers_
	center_colors = colors[:len(centers)]

	return X, y_pred, X_pca[:,:2], params
