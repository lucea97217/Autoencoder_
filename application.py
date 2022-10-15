#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import pandas as pd
from sklearn import tree, datasets
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve, train_test_split

from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

SEED = np.random.seed(1)

#%%
# Q6. même question avec les données de reconnaissances de texte 'digits'

# Import the digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


train_dir = "train.csv"
test_dir = "test.csv"

df_train = pd.read_csv(train_dir)
y_train = df_train['label']
X_train = df_train.drop(columns=['label'])  # suppression du label

NUM_CAT_DIGIT = y_train.nunique()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

X_test = pd.read_csv(test_dir)


#%%

dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
dt.fit(X_train, y_train)

print(dt)
print(dt.get_params())
print(dt.feature_importances_)
#%%
plt.figure()
plt.imshow(dt.feature_importances_.reshape(28, 28))

#%%
# Q7. estimer la meilleur profondeur avec un cross_val_score

# Let's try to tune the `max_depth` hyperparameter using the cross validation with $5$-folds. For that we can use the `cross_val_score` function like `cross_val_score(clf, X, y, cv=5, n_jobs=-1)` and then plot the mean of the results for each depth-value tested. This lets us determine which depth could be better to use and then we can fit our tree and test it. This can become quite cumbersome as we can see below. 

ent_cv_err = []
gini_cv_err = []

for depth in range(1, 30):
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    acc = cross_val_score(dt, X_train, y_train, cv=5, n_jobs=-1)
    ent_cv_err.append((1 - acc.mean()) * 100)

    dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=depth)
    acc = cross_val_score(dt, X_train, y_train, cv=5, n_jobs=-1)
    gini_cv_err.append((1 - acc.mean()) * 100)


plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), ent_cv_err, label="entropy")
plt.plot(range(1, 30), gini_cv_err, label="gini")
plt.xlabel('Depth tree')
plt.ylabel("Error (in %)")
plt.legend()
plt.title("""Misclassification using 5-folds cross validation for the 
entropy and gini criteria by the depth of the tree""")
plt.axhline(y= min(ent_cv_err), color='r', linestyle='-')
plt.text(0, 14, '13.91%', color = 'red', fontsize=16)
plt.savefig("tree.jpg")
plt.show()
#dt_cv = tree.DecisionTreeClassifier(criterion='entropy', max_depth=np.argmin(ent_cv_err) + 1) 
#dt_cv.fit(X_val, y_val)
#print("The error using this method equals {:.3f}%.".format((1 - dt_cv.score(X_test, y_val)) * 100))

#%%
# So let's use the `GridSearchCV` function with a dictionnary to test different hyperparameters. Note that `n_jobs=-1` is used to accelerate the computation time.

parameters = {'max_depth': range(1, 50)}
grid_clf = GridSearchCV(tree.DecisionTreeClassifier(
    criterion="entropy", random_state=SEED), parameters, cv=5, n_jobs=-1)
grid_clf.fit(X_train, y_train)
best_tree = grid_clf.best_estimator_
print("The best depth is {} for an error of {:.3f}%.".format(
    grid_clf.best_params_["max_depth"], (1-best_tree.score(X_test, y_val)) * 100))

# Let's try to select another parameter with the depth.

#%%

parameters = {'max_depth': range(1, 30),
              'min_samples_split': np.arange(2, 50, 1)}

grid_clf2 = GridSearchCV(tree.DecisionTreeClassifier(
    criterion="entropy", random_state=SEED), parameters, cv=5, n_jobs=-1)

grid_clf2.fit(X_train, y_train)
best_tree = grid_clf2.best_estimator_
print("The best depth is {} associated with a minimal samples leaf of {:.3f}, for an error of {:.3f}%.".format(
    grid_clf2.best_params_["max_depth"],
    grid_clf2.best_params_["min_samples_split"], (1-best_tree.score(X_val, y_val)) * 100))

# So we can increase a little our accuracy tuning both the number of samples to make a split and the depth of the tree.
# Notice that the more parameters we want tun in with the `GridSearch`, the more time it takes to compute, and the gain
# might be (like in this case) very small.

## Question 8: Learning curve

# Let's look at the learning curve of a decision tree using the entropy criterion and the best depth determined
# previously (not the one with the `min_samples_split`).

#%%
clf = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=grid_clf.best_params_["max_depth"])

train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(best_tree, data, digits.target, cv=5, n_jobs=-1,
                   train_sizes=np.linspace(.1, 1, num=30, endpoint=True),
                   return_times=True, random_state=SEED)

#%%

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

#%%

_, axes = plt.subplots(1, 1, figsize=(8, 5))
plt.title("Learning curves on the dataset digits")
axes.set_xlabel("Number of samples")
axes.set_ylabel("Accuracy score")
axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="b")
axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
axes.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")
axes.legend(loc="best")
plt.tight_layout()
plt.show()
