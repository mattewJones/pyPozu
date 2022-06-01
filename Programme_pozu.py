from coord_corps_Mediapipe import *
from features import *
from load_db import *
from load_db import *
from preprocess import *
from SB_hist import *
from segmentation_Mediapipe import *
import sklearn.decomposition as dcp
import statistics
from sklearn.neighbors import KNeighborsClassifier
import cv2

def learn_from_all_data():
    
    # Chargement et pré-traitement de la base de donnée
    db_dir = Path("./DB_RESIZED")
    # prétraitement
    resize_whole_database(db_dir)
    # lecture
    
    db_data = read_database(db_dir)
    learn_data = db_data
    
    
    #print("ETAPE - CHARGEMENT ET PRE-TRAITEMENT TERMINEE")

    # Features
    labels=list(learn_data.keys())

    # calcul des attributs et mise sous forme exigée par sklearn
    learn_features,learn_labels=calc_feature_set(learn_data)
    
    
    #normalisation (apparemment ça permet à l'ACP de beaucoup mieux marcher)
    s=StandardScaler()
    s.fit(learn_features) #on ne normalise qu'avec les valeurs des données d'entraînement évidemment
    learn_features_norm = s.transform(learn_features)
    
    
    # Test Features
    PCA=dcp.PCA(.95)
    PCA.fit(learn_features_norm) #on ne calcule le projecteur qu'avec les données d'entrainement
    learn_features_PCA=PCA.transform(learn_features_norm)
    #print("PCA, learn_features_PCA, eval_features_PCA : ",PCA,learn_features_PCA, eval_features_PCA)
    print("FEATURES LOADED")  
        
    # Classification de l'image d'entrée
    clf = KNeighborsClassifier(p=1)
    clf.fit(learn_features_PCA,learn_labels)
    
    print("ETAPE - FIN learn from all data")
    
    return(s,clf,PCA)
    

def association_pozu(src,s,clf,PCA):
    # On redimensionne l'image et on la sauvegarde
    dest = "Image_reduite/image_reduite.jpg"
    resize_image(src,dest,(400,600))
    img= cv2.imread(dest)
    
    #print("ETAPE - IMAGE REDIMENSIONNEE")
    
    feature=extract_ACP_feature(img,PCA,s)
    #print("ETAPE - FEATURE OBTENUE")

    return feature

def labellisation(feature,clf):
    label=clf.predict(feature.reshape(1,-1))
    
   # print('label')
    return label
   
def jojo_pose(label):
    
    personnage = label[0]
   # print(personnage)
    path = 'DB_Anime/'+personnage+'.jpg'
    #print(path)
    perso_affiche = cv2.imread(path)
    cv2.imshow('YOU ARRRRRR : ',perso_affiche)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    

    
# s,clf,PCA=learn_from_all_data()

# feature = association_pozu("test6.jpg",s,clf,PCA)

# display_coords(cv2.imread("test6.jpg"))

# label = labellisation(feature)

# jojo_pose(label)