# Molecule-Classifier
Machine Learning / Data Mining - Classifier that implements Decision Trees and Naive Bias algorithms to Decide whether a molecule is 
active or not. Binary classification.  A molecule can be represented by several thousands of binary features which represent their topological shapes and other characteristics important for binding.

## Dataset
train - 800 records, max 100000 features, labels <br />
test - 350 records with features, no labels <br />

## My Approach 
Spare matrix to store data due to significant amount of zeros. TruncatedSVD (TSVD) to reduce features, SMOTE to help with skewed dataset.
Use Decision tree and Naïve Bayes - Gaussian to predict test labels.

## Libraries
numpy <br />
scipy <br />
sklearn <br />
  *sklearn.decomposition import TruncatedSVD
  sklearn.tree import DecisionTreeClassifier

