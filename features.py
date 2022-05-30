from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as dcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex,hsv_to_rgb
import pandas as pd
from pathlib import Path

# from load_db import learn_data,eval_data
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


def save_feature(features,labels,saveFile:Path):
	DF=pd.DataFrame(features,labels)
	DF.to_csv(saveFile)



def save_all_feature_data(learn_features,learn_labels,eval_features,eval_labels):
	"""
	enregistre les valeurs des attributs 
	dans un fichier csv
	"""
	save_feature(learn_features,learn_labels,Path("./learn.csv"))
	save_feature(eval_features,eval_labels,Path("./eval.csv"))

def load_feature(saveFile:Path):
	DF=pd.read_csv(saveFile,index_col=0)
	features=DF.to_numpy()
	labels=DF.index #oui j'utilise la table dans le mauvais sens et alors ?
	return features,labels

def load_all_feature_data():
	"""
	lit les valeurs des attributs 
	enregistrées dans un fichier csv
	"""
	learn_features,learn_labels=load_feature(Path("./learn.csv"))
	eval_features,eval_labels=load_feature(Path("./eval.csv"))
	return learn_features,learn_labels,eval_features,eval_labels


# #liste des étiquettes

# labels=list(learn_data.keys())

# # calcul des attributs et mise sous forme exigée par sklearn

# learn_features,learn_labels=calc_feature_set(learn_data)
# eval_features,eval_labels=calc_feature_set(eval_data)



# normalisation (apparemment ça permet à l'ACP de beaucoup mieux marcher)

# s=StandardScaler()
# s.fit(learn_features) #on ne normalise qu'avec les valeurs des données d'entraînement évidemment
# learn_features_norm = s.transform(learn_features)
# eval_features_norm = s.transform(eval_features)

def ACP_95(learn_features_norm,eval_features_norm):
    # ACP (avec 95% de la variance)
    PCA=dcp.PCA(.95)
    PCA.fit(learn_features_norm) #on ne calcule le projecteur qu'avec les données d'entrainement
    learn_features_PCA=PCA.transform(learn_features_norm)
    eval_features_PCA=PCA.transform(eval_features_norm)
    return (PCA,learn_features_PCA, eval_features_PCA)



def extract_ACP_feature(img,PCA,s):
	"""
	pour faire la classification d'autres images
	"""
    
	raw_feature=extract_feature(img)
	PCA_feature=PCA.transform(s.transform(raw_feature))
	return PCA_feature



def plotPCA(principalComponents_learn,principalComponents_eval,labels,learn_labels,eval_labels):
    teintes=np.linspace(0,0.8,len(labels))
    colors=[to_hex(hsv_to_rgb([t,1,1])) for t in teintes]
   	
    my_dpi = 96
    fig = plt.figure(figsize=(1000/my_dpi, 800/my_dpi), dpi=my_dpi)
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






def plotFeatureProjection(i,j):
	"""
	projection dans le plan U_i, U_j
	"""
	teintes=np.linspace(0,0.8,len(labels))
	colors=[to_hex(hsv_to_rgb([t,1,1])) for t in teintes]

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Axe U%d' % i )
	ax.set_ylabel('Axe U%d' % j )
	ax.set_title('attributs (normalisés) dans un plan')

	legendList=[]

	#plot learn features
	for color,label in zip(colors,labels) :
		indicesToKeep=np.where(learn_labels==label)
		ax.scatter(
			learn_features_norm[indicesToKeep,i],
			learn_features_norm[indicesToKeep,j],
			marker='^', c=color, s=100
		)
		legendList.append(label+" (learn)")

	#plot validation features
	for color,label in zip(colors,labels) :
		indicesToKeep=np.where(eval_labels==label)
		ax.scatter(
			eval_features_norm[indicesToKeep,i],
			eval_features_norm[indicesToKeep,j],
			marker='o', c=color, s=100
		)
		legendList.append(label+" (eval)")


	ax.legend(legendList)
	ax.grid()
	fig.show()
             


# if __name__=="__main__":
# 	#test de l'ACP : projection sur le sev de dimension 2
# 	#de plus grande inertie


# 	PCA_test=dcp.PCA(n_components=2)
# 	PCA_test.fit(learn_features_norm)
# 	principalComponents_learn = PCA_test.transform(learn_features_norm)
# 	principalComponents_eval=PCA_test.transform(eval_features_norm)

# 	plotPCA(principalComponents_learn,principalComponents_eval)