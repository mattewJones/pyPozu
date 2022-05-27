from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as dcp
import numpy as np
import matplotlib.pyplot as plt



from load_db import learn_data,eval_data
from calcul_distances import coord_corps,calcul_distances




def extract_feature(img):
	"""
	n'utilise que la pose pour l'instant
	"""
	coords=coord_corps(img)
	if len(coords)<33: 
		#33 : taille attendue de la sortie parce que des fois c'est moins hein parce qu'avoir un résultat cohérent c'est pour les pauvres hein librairie de merde langage de merde "we're beetwen consenting adults" TA GUEULE VAN ROSSUM 
		print("problèmes de détection sur 1 des images")
		coords=np.full(33,1)

	pose_feature=calcul_distances(coords)


	return(np.array(pose_feature))



def calc_feature_set(dataset_dict):
	"""
	transforme les données d'échantillons indexées par
	classes dans dataset_dict en un tableau de valeurs d'attributs
	et un tableau contenant les étiquettes 
	des classes correspondantes (format exigé par sklearn)
	"""
	resFeatures=[]
	resLabels=[]
	for label,sampleList in dataset_dict.items():
		for sample in sampleList :
			feature=extract_feature(sample.imData())
			resFeatures.append(feature)
			resLabels.append(label)
	return(np.array(resFeatures),np.array(resLabels))


#liste des étiquettes

labels=list(learn_data.keys())


# calcul des attributs et mise sous forme exigée par sklearn

learn_features,learn_labels=calc_feature_set(learn_data)
eval_features,eval_labels=calc_feature_set(eval_data)



# normalisation (apparemment ça permet à l'ACP de beaucoup mieux marcher)

s=StandardScaler()
s.fit(learn_features) #on ne normalise qu'avec les valeurs des données d'entraînement évidemment
learn_features_norm = s.transform(learn_features)
eval_features_norm = s.transform(eval_features)


# ACP (avec 95% de la variance)

PCA=dcp.PCA(.95)
PCA.fit(learn_features_norm) #on ne calcule le projecteur qu'avec les données d'entrainement
learn_features_PCA=PCA.transform(learn_features_norm)
eval_features_PCA=PCA.transform(eval_features_norm)



def extract_ACP_feature(img):
	"""
	pour faire la classification d'autres images
	"""
	raw_feature=extract_feature(img)
	PCA_feature=PCA.transform(s.transform(raw_feature))
	return PCA_feature



if __name__=="__main__":
	#test de l'ACP : projection sur le sev de dimension 2
	#de plus grande inertie


	PCA_test=dcp.PCA(n_components=2)
	principalComponents = PCA_test.fit_transform(learn_features_norm)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1')
	ax.set_ylabel('Principal Component 2')
	ax.set_title('2 component PCA')


	for label in labels :
		indicesToKeep=np.where(learn_labels==label)
		ax.scatter(
			principalComponents[indicesToKeep,0],
			principalComponents[indicesToKeep,1],
			s = 50
		)

	ax.legend(labels)
	ax.grid()
	fig.show()