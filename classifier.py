from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

# from features import learn_features_PCA,learn_labels,learn_data
# from features import eval_features_PCA,eval_labels,labels


# Classifieur
def classification(learn_features_PCA,learn_labels,eval_features_PCA):
    clf = KNeighborsClassifier(p=1)
    clf.fit(learn_features_PCA,learn_labels)
    predicted_labels=clf.predict(eval_features_PCA)
    return predicted_labels




# Matrice de confusion
def calcul_mat_conf(eval_labels,predicted_labels,labels):
    conf_mat=confusion_matrix(y_true=eval_labels,y_pred=predicted_labels,labels=labels)
    return conf_mat



# F1 Score et tout
def evaluation_score(eval_labels,predicted_labels,labels):
    prec,recll,fscore,spp=precision_recall_fscore_support(
    	y_true=eval_labels,
    	y_pred=predicted_labels,
    	labels=labels
    	)
    return (prec,recll,fscore,spp)




# print(conf_mat)

# print('true class : ', eval_labels)
# print('result     : ', predicted_labels)
# print(fscore)