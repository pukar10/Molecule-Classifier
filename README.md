# Molecule-Classifier <br /n>
Machine Learning / Data Mining - Classifier that implements Decision Trees and Naive Bias algorithms to Decide whether a molecule is 
active or not. Binary classification.  A molecule can be represented by several thousands of binary features which represent their topological shapes and other characteristics important for binding.

## Dataset <br /n>
train - 800 records, max 100000 features, labels <br /n>
test - 350 records with features, no labels <br /n>

## My Approach <br /n>
Spare matrix to store data due to significant amount of zeros. TruncatedSVD (TSVD) to reduce features, SMOTE to help with skewed dataset.
Use Decision tree and Naïve Bayes - Gaussian to predict test labels.

## Libraries <br /n>
numpy <br /n>
scipy <br /n>
sklearn <br /n>
  *sklearn.naive_bayes import GaussianNB <br /n>
  *sklearn.neural_network import MLPClassifier <br /n>
  *sklearn.decomposition import TruncatedSVD <br /n>
  *sklearn.tree import DecisionTreeClassifier <br /n>
imblearn.over_sampling import SMOTE <br /n>
