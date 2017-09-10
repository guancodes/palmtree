from palmtree import data, model
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import accuracy_scorer
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
np.random.seed(1)


print("loading data ...")
X, y, _ = data.load_data()

print("creating model ...")
#mod = model.make_model(LinearDiscriminantAnalysis)
mod = model.make_model(SVC)
#mod = model.make_model(MLPClassifier, solver='lbfgs', random_state=1)

print("training model ...")

# LDA
#params = {'model__tol': [1e-4]}

# SVC 
params = {'model__C': [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], 
          'model__gamma': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}

# MLP
# mlp_grid = []
# for i in range(1, int(X.shape[1] / 2)):
#     for j in range(1, int(X.shape[1] / 2)):
#         mlp_grid.append((i, j))
# params = {'model__hidden_layer_sizes': mlp_grid}

gcv = GridSearchCV(mod, params, scoring=accuracy_scorer, cv=10, n_jobs=4)
gcv.fit(X, y)
print("model score =", gcv.best_score_)
print("model params =", gcv.best_params_)

print("saving model ...")
mod = gcv.best_estimator_
mod.named_steps['feature_extractor'].active = True
model.save_model(mod, "wicked.mod")

print("done.")
