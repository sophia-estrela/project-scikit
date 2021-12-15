from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

# max_depth=-10 fails during fit
param_grid = {"max_depth": [-10, 3, 10]}
search = HalvingGridSearchCV(
    clf,
    param_grid,
    resource="n_estimators",
    max_resources=10,
    random_state=0,
    refit=False,
)
search.fit(X, y)

print("-------------BEST SCORE:", search.best_score_)
print("-------------BEST PARAMS:", search.best_params_)