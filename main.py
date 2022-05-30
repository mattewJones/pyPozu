from calcul_distances import *
from classifier import *
from coord_corps_Mediapipe import *
from features import *
from load_db import *
from load_db import *
from preprocess import *
from SB_hist import *
from segmentation_Mediapipe import *

def main():

    ## Chargement et pré-traitement de la base de donnée ###########
    db_dir = Path("./DB_RESIZED")
    # prétraitement
    resize_whole_database(db_dir)
    # lecture
    db_data = read_database(db_dir)
    learn_data, eval_data = split_database(db_data, 0.75)
    
    print("ETAPE - CHARGEMENT ET PRE-TRAITEMENT TERMINEE")

    ## Attributs ############

    # nom des classes
    labels=list(learn_data.keys())

    # calcul des attributs et mise sous forme exigée par sklearn
    learn_features,learn_labels=calc_feature_set(learn_data)
    eval_features,eval_labels=calc_feature_set(eval_data)

    # enregistrement
    save_all_feature_data(learn_features,learn_labels,eval_features,eval_labels)
    
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
    
    print("Matrice de confusion : ")
    print(conf_mat)
    print('\n\ntrue class : ', eval_labels)
    print('\n\n result     : ', predicted_labels)
    print(fscore)
    print("ETAPE - EVALUATION TERMINEE")
    
    
main()