from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


from load_db import learn_data,eval_data



def extract_feature(img):
	"""
	pour tester
	"""
	return([1,2,3]+0.1*np.random.rand(3))



def calc_feature_set(dataset_dict):
	"""
	transforme les données d'échantillons indexées par
	classes dans dataset_dict en un tableau de valeurs d'attributs
	et un tableau contenant les étiquettes 
	des classes correspondantes
	"""
	resFeatures=[]
	resLabels=[]
	for label,sampleList in dataset_dict.items():
		for sample in sampleList :
			feature=extract_feature(sample.imData())
			resFeatures.append(feature)
			resLabels.append(label)
	return(np.array(resFeatures),np.array(resLabels))


# calcul des attributs et mise sous forme exigée par sklearn

learn_features,learn_labels=calc_feature_set(learn_data)
eval_features,eval_labels=calc_feature_set(eval_data)



# normalisation (apparemment ça permet à l'ACP de beaucoup mieux marcher)

learn_features_norm = StandardScaler().fit_transform(learn_features)
eval_features_norm = StandardScaler().fit_transform(eval_features)



# ACP

PCA_transform=PCA(n_components=2)
principalComponents = PCA_transform.fit_transform(learn_features_norm)


