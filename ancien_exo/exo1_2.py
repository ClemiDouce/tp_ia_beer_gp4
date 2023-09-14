from sklearn import datasets
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

# Exercice 1
# 1. 
X,y = datasets.make_classification(200, n_features=2, n_redundant=0) 

# 2. 3.
class_un = X[y == 1]
class_zero = X[y == 0]

fig, ax = plt.subplots()
ax.grid(linestyle=":")
ax.plot(class_zero, "ro")
ax.plot(class_un, "bo")
fig.savefig("./output/figure_classification.jpg")

# Exercice 2
# 1.
log_reg = linear_model.LogisticRegression(penalty=None)
log_reg.fit(X, y)

# 4.
random_points = np.random.rand(100, 2) * 8 -4
predicts = log_reg.predict(random_points)
print(predicts)
class_un_predict = random_points[predicts == 1]
class_zero_predict = random_points[predicts == 0]




fig, ax = plt.subplots()
ax.grid(linestyle=":")
ax.plot(class_zero[:,0], class_zero[:,1], "ro")
ax.plot(class_un[:,0], class_un[:,1], "bo")
ax.plot(class_un_predict[:,0], class_un_predict[:, 1], "co")
ax.plot(class_zero_predict[:,0], class_zero_predict[:, 1], "yo")
fig.savefig("./output/figure_classification_predict.jpg")
plt.show()

DecisionBoundaryDisplay.from_estimator(
    log_reg,
    X,
    ax=ax,
    response_method="predict",
    grid_resolution=200,
    cmap=plt.cm.Paired,
)
fig.savefig("./output/borne_decision.png")





##import matplotlib . pyplot as plt 
# fig, ax=plt.subplots()
#a x . g r i d ( l i n e s t y l e =’ - - ’ )
# "ro" signifie "rouge" + "gros points"
#ax.plot([1, 2, 3, 4], [2, 4, 3, 3], ’ro’) plt .show()
# Sauvegarde la figure si besoin # fig.savefig("figuretest.png")