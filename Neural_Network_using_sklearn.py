from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier,SGDRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

data = sio.loadmat('ex3data1.mat')

A = data['X']
B = data['y']

rkf = model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
for train_index,test_index in rkf.split(A):
    X_train, X_test = A[train_index], A[test_index]
    y_train, y_test = B[train_index], B[test_index]
#X_train,X_test,y_train,y_test = train_test_split(A,B,test_size=0.3,random_state = 42)

scaler=StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True,with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(25),max_iter=500)
clf.fit(X_train,y_train) 
p=clf.predict(X_test)
print('This is the confusion matrix')
print(confusion_matrix(y_test,p))
print('This is the classification report')
print(classification_report(y_test,p))
print('The accuracy obtained using sklearn implementation')
print(clf.score(X_test,y_test)*100)

 
