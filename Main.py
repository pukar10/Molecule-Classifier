import os
import numpy as np
from sklearn import tree
import scipy.sparse as sps
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier


def main():
    molecules = np.zeros((800, 100001))
    isActive = np.zeros(800)

    # Open file
    n = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "train drugs.dat")
    with open(lines, "r") as f:
        for line in f:  # put 1 or 0 in isActive
            line = line.split()
            np.put(isActive, n, int(line[0]))  # replace line with 1
            del line[0]
            for i in line:  # put 1 or 0 in molecules, track if feature active
                molecules[n][int(i)] = 1
            n += 1

    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "train drugs.dat")
    with open(lines, "r") as f:
        for line in f:
            line = line.split()
            for i in line:


    spMatrix = sps.csr_matrix(molecules)

    TSVD = TruncatedSVD(n_components=100)
    uspMatrix = TSVD.fit(spMatrix).transform(spMatrix)
    print('Original number of features:', spMatrix.shape[1])
    print('Reduced number of features:', uspMatrix.shape[1])

    # decision tree model
    clf = DecisionTreeClassifier()
    clf = clf.fit(uspMatrix, isActive)
    print(clf)


if __name__ == "__main__":
    main()
