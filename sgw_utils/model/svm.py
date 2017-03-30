import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV


class SVM():
    """Wrapper on sk-learn based NearestNeighbor model. 
    """


    def __init__(self, kernel_type="linear", tol=0.00001, max_iter=10000, grid_search=True):

        self.kernel_type = kernel_type
        self.tol = tol
        self.max_iter = max_iter
        self.param_grid = [ 
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
        ]

        if grid_search:
            self.clf = GridSearchCV( svm.SVC(kernel=kernel_type, tol=tol, max_iter = max_iter, probability=True), 
                self.param_grid, n_jobs=-1 )
        else:
            self.clf = svm.SVC(kernel=kernel_type, tol=tol, max_iter = max_iter, probability=True)

    def __str__(self):
        return vars(self)

    def fit(self, X, y):
        self.clf.fit(X, y)


    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def load(self, filename):
        try:
            self.clf = joblib.load(filename)
        except IOError, e:
            print "All the pkl files are not found in the same directory. "
            raise e

    def save(self, filename):
        outfiles = joblib.dump(self.clf, filename)
        print "Saved SVM model as: "
        for ofn in outfiles:
            print ofn
        print "Note: All pkl files are required in the same folder when reloading the model. "


if __name__ == "__main__":
    classifier = SVM()