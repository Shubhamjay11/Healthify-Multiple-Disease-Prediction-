from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_jobs=-1, verbose=2),
    "log_reg": linear_model.LogisticRegression(),
    "decision_tree": tree.DecisionTreeClassifier(),
}
