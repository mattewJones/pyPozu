from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

from features import learn_features_PCA,learn_labels,learn_data
from features import eval_features_PCA,eval_labels,labels








clf = KNeighborsClassifier(p=1)
clf.fit(learn_features_PCA,learn_labels)
predicted_labels=clf.predict(eval_features_PCA)





conf_mat=confusion_matrix(
	y_true=eval_labels,
	y_pred=predicted_labels,
	labels=labels
	)




prec,recll,fscore,spp=precision_recall_fscore_support(
	y_true=eval_labels,
	y_pred=predicted_labels,
	labels=labels
	)




print(conf_mat)

print(eval_labels)
print(predicted_labels)
print(fscore)