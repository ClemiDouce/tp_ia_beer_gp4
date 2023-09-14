from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

# 1.
X, y = make_circles(200, noise=0.2, random_state=42)

class_zero = X[y == 0]
class_un = X[y == 1]

fig, ax = plt.subplots()
ax.grid(linestyle=":")
ax.plot(class_zero[:, 0], class_zero[:, 1], "ro")
ax.plot(class_un[:, 0], class_un[:, 1], "bo")
fig.savefig("./output/figure_circle.jpg")


# 2.
r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
theta = np.arctan2(X[:, 1], X[:, 0])

X_polar = np.column_stack((r, theta))

h = 0.02  # Pas pour la grille de décision

# Création du classifier avec la standardisation
classifier = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver="lbfgs",
        random_state=1,
        max_iter=2000,
        early_stopping=True,
        hidden_layer_sizes=[10, 10],
    ),
)

# Création de la figure
figure = plt.figure()

# Calcul des limites du graphique
x_min, x_max = X_polar[:, 0].min() - 0.5, X_polar[:, 0].max() + 0.5
y_min, y_max = X_polar[:, 1].min() - 0.5, X_polar[:, 1].max() + 0.5

# Création de la grille à l'aide des limites et du pas
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Entraînement du modèle sur les données
classifier.fit(X_polar, y)

# Calcul des prédictions sur la grille
if hasattr(classifier, "decision_function"):
    Z = classifier.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
else:
    Z = classifier.predict_proba(np.column_stack((xx.ravel(), yy.ravel())))[:, 1]

Z = Z.reshape(xx.shape)

# Tracé de la frontière de décision et des points de données
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF4500", "#0000CC"])
ax = plt.subplot()
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
ax.scatter(X_polar[:, 0], X_polar[:, 1], c=y, cmap=cm_bright, edgecolors="k")

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# Sauvegarder la figure
figure.savefig("./output/decision_polar_standard_scaler.png")
