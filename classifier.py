from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from load_db import learn_data,eval_data
from calc_features import calc_feature_vector
import numpy as np



def calc_feature_array(dataset_dict):
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
			feature=calc_feature_vector(sample.imData())
			resFeatures.append(feature)
			resLabels.append(label)
	return(np.array(resFeatures),np.array(resLabels))





learn_features,learn_labels=calc_feature_array(learn_data)
eval_features,eval_labels=calc_feature_array(eval_data)

clf = KNeighborsClassifier(p=1)
clf.fit(learn_features,learn_labels)
predicted_labels=clf.predict(eval_features)

print(predicted_labels)

labels=list(learn_data.keys())

print(
	confusion_matrix(
		y_true=eval_labels,
		y_pred=predicted_labels,
		labels=labels
		)
	)


print(
	np.array(precision_recall_fscore_support(
		y_true=eval_labels,
		y_pred=predicted_labels,
		labels=labels
		)
	)
)




print(labels)


