import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib


class NN():
    """Wrapper on sk-learn based NearestNeighbor model. 
    """

    def __init__(self, n_neighbors=1, algorithm='ball_tree'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm

        self.clf = NearestNeighbors(n_neighbors, algorithm)
        self.index_table = None

    def __str__(self):
        return vars(self)

    def fit(self, X):
        self.clf.fit(X)

    def predict(self, X, index_table=None, k=0, dist_return=False):
        if index_table is None:
            index_table = self.index_table
        dist, indices = self.clf.kneighbors(X)
        indices = indices[:,k]
        if dist_return:
            return dist[:,k], index_table[indices]
        else: 
            return index_table[indices]

    def predict_all(self, X, index_table=None, dist_return=False):
        if index_table is None:
            index_table = self.index_table
        dist, indices = self.clf.kneighbors(X)
        if dist_return:
            return dist, index_table[indices]
        else: 
            return index_table[indices]

    def load(self, filename):
        try:
            self.clf = joblib.load(filename)
        except IOError, e:
            print "Not found all pkl files in workspace. "
            raise e
        index_name = os.path.join(os.path.dirname(filename), "nn_index_table.npy")
        self.index_table = np.load(index_name)
        return self.index_table

    def save(self, filename, y):
        outfiles = joblib.dump(self.clf, filename)
        print "Saved NN model as :"
        for ofn in outfiles:
            print ofn
        print "Note: All pkl files are required in the same folder when reloading the model. "
        ofn_index_table = "nn_index_table.npy"
        print "Additionally saved index table as '%s' " % ofn_index_table
        np.save(os.path.join(os.path.dirname(filename), ofn_index_table), y)


if __name__ == "__main__":
    classifier = NN()