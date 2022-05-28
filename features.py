from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as dcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex,hsv_to_rgb


from load_db import learn_data,eval_data
from calcul_distances import coord_corps,calcul_distances
from SB_hist import SB_hist




def extract_feature(img):
	"""
	utilise pose et histogramme
	"""
	coords=coord_corps(img)
	pose=np.array(calcul_distances(coords))

	hist=SB_hist(img)

	feature=np.concatenate([pose,hist])

	return(feature)



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
			try:
				feature=extract_feature(sample.imData())
			except:
				feature=np.zeros(72) #72 : taille normale de l'attribut
				print('erreur d\'extraction des attributs sur : ',sample.adress)

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
	PCA_test.fit(learn_features_norm)
	principalComponents_learn = PCA_test.transform(learn_features_norm)
	principalComponents_eval=PCA_test.transform(eval_features_norm)


	def plotPCA(principalComponents_learn,principalComponents_eval):
		teintes=np.linspace(0,0.8,len(labels))
		colors=[to_hex(hsv_to_rgb([t,1,1])) for t in teintes]
		
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 1')
		ax.set_ylabel('Principal Component 2')
		ax.set_title('2 component PCA')

		legendList=[]

		#plot learn features
		for color,label in zip(colors,labels) :
			indicesToKeep=np.where(learn_labels==label)
			ax.scatter(
				principalComponents_learn[indicesToKeep,0],
				principalComponents_learn[indicesToKeep,1],
				marker='^', c=color, s=100
			)
			legendList.append(label+" (learn)")

		#plot validation features
		for color,label in zip(colors,labels) :
			indicesToKeep=np.where(eval_labels==label)
			ax.scatter(
				principalComponents_eval[indicesToKeep,0],
				principalComponents_eval[indicesToKeep,1],
				marker='o', c=color, s=100
			)
			legendList.append(label+" (eval)")


		ax.legend(legendList)
		ax.grid()
		fig.show()

	plotPCA(principalComponents_learn,principalComponents_eval)