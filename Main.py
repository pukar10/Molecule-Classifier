from scipy import sparse
from sklearn import tree
import os
import numpy as np
import scipy.sparse as sps
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def main():
    # Create all numpy arrays
    testMoleculeFeatures = np.zeros((800, 100001))
    trainMoleculeFeatures = np.zeros((800, 100001))
    trainClassLabels = np.zeros(800)

    # Open training data file
    n = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "train drugs.dat")
    with open(lines, "r") as f:
        for line in f:  # put 1 or 0 in trainClassLabels
            line = line.split()
            np.put(trainClassLabels, n, int(line[0]))  # replace line with 1
            del line[0]
            for i in line:  # put 1 or 0 in trainMoleculeFeatures, track if feature active
                trainMoleculeFeatures[n][int(i)] = 1
            n += 1
    print(trainMoleculeFeatures[0][190])
    print(trainMoleculeFeatures[0][191])
    print(trainMoleculeFeatures[0][192])
    # Training data: Sparse Matrix & reduce features
    print(len(trainMoleculeFeatures))
    print(len(trainMoleculeFeatures[0]))

    trainspMatrix = sps.csr_matrix(trainMoleculeFeatures)

    TSVD = TruncatedSVD(n_components=1000)  # try higher number
    trainuspMatrix = TSVD.fit_transform(trainspMatrix)
    # trainuspMatrix = TSVD.fit(trainspMatrix, trainClassLabels)
    # trainuspMatrix = trainuspMatrix.transform(trainspMatrix)

    print(len(trainMoleculeFeatures))
    print(len(trainMoleculeFeatures[0]))

    trainuspMatrix = TSVD.fit_transform(trainuspMatrix)
    print('Original number of features:', trainuspMatrix.shape[1])
    print('Reduced number of features:', trainuspMatrix.shape[1])

    # Training Data Oversampling
    print("Oversampling using SMOTE")
    # random_state = 10
    smote = SMOTE()
    trainuspMatrix, trainClassLabels = smote.fit_sample(trainuspMatrix, trainClassLabels)

    # open testing file data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "test.dat")
    n = 0
    with open(lines, "r") as f:
        for line in f:
            line = line.split()
            for i in line:
                testMoleculeFeatures[n][int(i)] = 1
            n += 1

    # Reduce Test data Features
    # testReducedMoleculeFeatures = TSVD.fit(testMoleculeFeatures).transform(testMoleculeFeatures)
    testReducedMoleculeFeatures = TSVD.fit_transform(testMoleculeFeatures)

    # Classifying
    # criterion='gini'
    # min_samples_leaf = -
    # max_depth=8 |
    classifiers = [DecisionTreeClassifier(max_depth=8)]
    classifiers = classifiers.fit(trainuspMatrix, trainClassLabels)

    # Predict Class Labels
    testMoleculePredictedClasses = classifiers.predict(testReducedMoleculeFeatures)
    output = open('RESULTS.txt', 'w')
    for i in testMoleculePredictedClasses:
        if int(i) is -1:
            i = 0
        output.write(str(i) + "\n")
    output.close()

    print("Done!")


if __name__ == "__main__":
    main()
