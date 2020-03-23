from scipy import sparse
from sklearn import tree
import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def main():
    start = time.time()
    # Create all numpy arrays
    # testMoleculeFeatures = np.zeros((800, 100001))
    # trainMoleculeFeatures = np.zeros((800, 100001))
    # trainClassLabels = np.zeros(800)
    testMoleculeFeatures = []
    trainMoleculeFeatures = []
    trainClassLabels = []

    print("Arrays initialized")

    # Open training data file
    n = 0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "train drugs.dat")
    with open(lines, "r") as f:
        for line in f:  # put 1 or 0 in trainClassLabels
            temp = [0 for i in range(100001)]
            # temp = np.zeros(100001)
            line = line.split()
            # np.put(trainClassLabels, n, line[0])
            trainClassLabels.append(line[0])
            del line[0]
            for i in line:
                # np.put(temp, int(i), 1)
                temp[int(i)] = 1
            trainMoleculeFeatures.append(temp)
            # np.put(trainMoleculeFeatures, n, temp)
            n += 1

    print('Train - Features and Labels obtained')
    print(len(trainMoleculeFeatures))
    print(len(trainMoleculeFeatures[0]))

    # Make numpy array and Sparse matrix
    trainMoleculeFeaturesNumpy = np.array(trainMoleculeFeatures)
    print('Numpy conversion finished')
    print(len(trainMoleculeFeatures))
    print(len(trainMoleculeFeatures[0]))
    trainspMatrix = csr_matrix(trainMoleculeFeaturesNumpy)
    print('Sparse matrix finished')

    # reduce features
    TSVD = TruncatedSVD(n_components=50)  # try higher number | reduced number of features
    trainReducedFeatures = TSVD.fit(trainspMatrix, trainClassLabels)
    print("Fit finished!")
    trainReducedFeatures = trainReducedFeatures.transform(trainspMatrix)
    # trainReducedFeatures = TSVD.fit_transform(trainspMatrix, trainClassLabels)
    print('Transform finished!')

    print('Original number of features:', trainspMatrix.shape[1])
    print('Reduced number of features:', trainReducedFeatures.shape[1])

    # Training Data Oversampling
    # random_state = 10
    smote = SMOTE()
    trainReducedFeatures, trainClassLabels = smote.fit_sample(trainReducedFeatures, trainClassLabels)
    print("Oversampling using SMOTE")

    # open testing file data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    lines = os.path.join(dir_path, "test.dat")
    n = 0
    with open(lines, "r") as f:
        for line in f:
            temp = [0 for i in range(100001)]
            # temp = np.zeros(100001)
            line = line.split()
            for i in line:
                # np.put(temp, int(i), 1)
                temp[int(i)] = 1
            # np.put(trainMoleculeFeatures, n, temp)
            testMoleculeFeatures.append(temp)
            n += 1

    print("Testing features Obtained")

    # Reduce Test data Features
    # don't fit test data because it is already fitted to your model or training data.
    testReducedMoleculeFeatures = TSVD.transform(testMoleculeFeatures)
    print("Test transform finished")

    # Classifying

    # Using Naive bayes Gaussian

    # classi = GaussianNB()
    # classi.fit(trainReducedFeatures, trainClassLabels)
    # print("Gaussian Naive Bayes Classifier created")
    #
    # # Predict Class Labels
    # testMoleculePredictedClasses = classi.predict(testReducedMoleculeFeatures)
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # textFile = os.path.join(dir_path, "Results.txt")
    # output = open(textFile, 'w')
    # for i in testMoleculePredictedClasses:
    #     if int(i) is -1:
    #         i = 0
    #     output.write(str(i) + "\n")
    # output.close()

    # Using Decision Tree
    # best = max_depth=8, min_samples_split=5, min_samples_leaf=5 -> .73
    # max_depth = 8 -> .73
    # max_depth = 10 -> .70
    # max_depth = 6 -> .71
    # NO min_split and min_leaf -> .65
    # max_depth = 8, min_samples_split = 4, min_samples_leaf = 4 -> .71
    # max_depth=8, min_samples_split=6, min_samples_leaf=6 -> .43
    classi = DecisionTreeClassifier(max_depth=8, min_samples_split=5, min_samples_leaf=5)
    classi = classi.fit(trainReducedFeatures, trainClassLabels)
    print("Decision tree Classifiers created")

    # Predict Class Labels
    testMoleculePredictedClasses = classi.predict(testReducedMoleculeFeatures)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    textFile = os.path.join(dir_path, "Results.txt")
    output = open(textFile, 'w')
    for i in testMoleculePredictedClasses:
        if int(i) is -1:
            i = 0
        output.write(str(i) + "\n")
    output.close()

    print("Predictions finished")
    print("RESULTS.txt created")
    print("Done!")

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()

# Decision tree using n_components = 50 is .73 accuracy
# best = max_depth=8, min_samples_split=5, min_samples_leaf=5 -> .73
    # max_depth = 8 -> .73
    # max_depth = 10 -> .70
    # max_depth = 6 -> .71
    # NO min_split and min_leaf -> .65
    # max_depth = 8, min_samples_split = 4, min_samples_leaf = 4 -> .71
    # max_depth=8, min_samples_split=6, min_samples_leaf=6 -> .43

# Naive Bayes using Guassian gave result of .59

