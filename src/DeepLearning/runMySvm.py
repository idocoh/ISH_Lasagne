# http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
print clf.fit(X, y)  
