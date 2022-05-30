from numpy.linalg import norm

from calcul_distances import *
from classifier import *
from coord_corps_Mediapipe import *
from features import *
from load_db import *
from load_db import *
from preprocess import *
from SB_hist import *
from segmentation_Mediapipe import *
from barycenters import calc_feature_bars




## Chargement et pré-traitement de la base de donnée ###########
db_dir = Path("./DB_RESIZED")
# prétraitement
resize_whole_database(db_dir)
# lecture
db_data = read_database(db_dir)
learn_data, eval_data = split_database(db_data, 0.75)

# nom des classes
labels=list(learn_data.keys())

print("ETAPE - CHARGEMENT ET PRE-TRAITEMENT TERMINEE")


## Attributs ############

# drapeau de recalcul des attributs
feature_calc_flag=False

if feature_calc_flag :

    # calcul des attributs et mise sous forme exigée par sklearn
    learn_features,learn_labels=calc_feature_set(learn_data)
    eval_features,eval_labels=calc_feature_set(eval_data)

    # enregistrement
    save_all_feature_data(learn_features,learn_labels,eval_features,eval_labels)

else :
    learn_features,learn_labels,eval_features,eval_labels=load_all_feature_data()

    
#normalisation (apparemment ça permet à l'ACP de beaucoup mieux marcher)
s=StandardScaler()
s.fit(learn_features) #on ne normalise qu'avec les valeurs des données d'entraînement évidemment
learn_features_norm = s.transform(learn_features)
eval_features_norm = s.transform(eval_features)

# PCA
PCA,learn_features_PCA, eval_features_PCA = ACP_95(learn_features_norm,eval_features_norm)
PCA_test=dcp.PCA(n_components=2)

# visualisation des attributs à l'aide d'une ACP
PCA_test.fit(learn_features_norm)
principalComponents_learn = PCA_test.transform(learn_features_norm)
principalComponents_eval=PCA_test.transform(eval_features_norm)
plotPCA(principalComponents_learn,principalComponents_eval,labels,learn_labels,eval_labels)
print("ETAPE - FEATURES TERMINEE")


## Evaluation ##########

predicted_labels = classification(learn_features_PCA,learn_labels,eval_features_PCA)
conf_mat = calcul_mat_conf(eval_labels,predicted_labels,labels)
prec,recll,fscore,spp = evaluation_score(eval_labels,predicted_labels,labels)

## Barycentres des attributs
bars=calc_feature_bars(learn_features_PCA,learn_labels,labels)
barDists=np.array([[norm(i-j) for i in bars] for j in bars])

print('\nBarycenters distances : ')
print(np.floor(barDists)) 
print("\nMatrice de confusion : ")
print(conf_mat)
print('\nvraie classe : ')
print(eval_labels)
print('\nresultat     : ')
print(predicted_labels)
print('\nF1 de chaque classe : ')
print(fscore)
print('\nF1 global : ')
print(np.mean(fscore))



print("ETAPE - EVALUATION TERMINEE")

