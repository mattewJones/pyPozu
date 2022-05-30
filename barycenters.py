from sklearn.cluster import KMeans
import numpy as np

def calc_feature_bars(learn_features_PCA,learn_labels,labels):
	res=[]
	for label in labels :
		indicesToKeep=np.where(learn_labels==label)
		kmeansEst=KMeans(n_clusters=1)
		kmeansEst.fit(learn_features_PCA[indicesToKeep,:][0])
		res.append(kmeansEst.cluster_centers_[0])
	return np.array(res)
