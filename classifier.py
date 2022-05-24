from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from load_db import learn_data,eval_data
from calc_features import calc_features 
import numpy as np



def calc_features(dataset_dict):
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
			feature=calc_features(sample.imData())
			resFeatures.append(feature)
			resLabels.append(label)
	return(np.array(resFeatures),np.array(resLabels))





learn_features,learn_labels=calc_features(learn_data)
eval_features,eval_labels=calc_features(eval_data)

clf = KNeighborsClassifier(p=1)
clf.fit(learn_features,learn_labels)
predicted_labels=clf.predict(eval_features)

print(predicted_labels)



print(confusion_matrix(y_true=eval_labels,y_pred=predicted_labels))