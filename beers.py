import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./recipeData.csv", encoding="latin1")
orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)
ser = df.isna().mean() * 100

df.to_csv("./out.csv")

df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[df["IBU"] <= 150]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

df['Bitterness_PA'] = df['IBU'] / df['ABV']  # Amertume par unité d'alcool 
df['Total_Density'] = df['OG'] - df['FG'] # Concentration de sucre dans la bière
df['OG_FG_Ratio'] = df['OG'] / df['FG'] # Taux de fermentation d'une bière

new_len = len(df)

one_hot = pd.get_dummies(df["BrewMethod"])
#print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)
columns_to_bin = ["Size(L)", "OG", "FG", "Color", "BoilSize"]
for column in columns_to_bin:
    df[f"{column}_Binned"] = pd.qcut(df[column], q=5, labels=False, duplicates="drop")


##### SCORE DE PREDICTION #####
X = df["OG"].values.reshape(-1, 1)
y = df["ABV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("Score de prédiction :", score)

##### EVALUATION DU MODELE (MSE) #####
# X, y = df["OG"].values.reshape(-1, 1), df["IBU"]
# X2 = df["OG"].values.reshape(-1, 1)
# linear = linear_model.LinearRegression(fit_intercept=False)

# linear.fit(X, y)

# predictions = linear.predict(X2)

# y = df["IBU"]

# y_pred = predictions

# mse = np.mean((y - y_pred) ** 2)

# print("Mean Squared Error (MSE) :", mse)

# plt.scatter(X, y, label='Données originales')
# plt.plot(x_pred, y_pred, color='red', label='Régression polynomiale')
# plt.xlabel('BoilTime')
# plt.ylabel('ABV')
# plt.legend()
# plt.savefig("./predictionNLpolynomial.png")

##### MATRICE DE CORRELATION #####
# plt.matshow(df.corr())
# plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

##### SCATTER #####
# fig, ax = plt.subplots()
# cm_bright = ListedColormap(["#FF4500", "#0000CC"])
# ax.scatter(df[df.columns[4]], df[df.columns[9]], cmap=cm_bright, edgecolors="k")
# fig.savefig("./rapport/ABV-BoilGravity.png")

##### PREDICTION LINEAIRE ABV #####
# X, y = df["OG"].values.reshape(-1, 1), df["ABV"]
# X2 = df["OG"].values.reshape(-1, 1)

# linear = linear_model.LinearRegression(fit_intercept=False)

# linear.fit(X, y)

# predictions = linear.predict(X2)

# print(predictions)

# plt.scatter(X2, predictions, edgecolors="k")
# plt.xlabel("Prédictions")
# plt.ylabel("Vraies valeurs (ABV)")
# plt.savefig("./predictionABV.png")
# plt.show()

##### PREDICTION LINEAIRE IBU #####
# X, y = df["OG"].values.reshape(-1, 1), df["IBU"]
# X2 = df["OG"].values.reshape(-1, 1)

# linear = linear_model.LinearRegression(fit_intercept=False)

# linear.fit(X, y)

# predictions = linear.predict(X2)

# print(predictions)

# plt.scatter(X2, predictions, edgecolors="k")
# plt.xlabel("Prédictions")
# plt.ylabel("Vraies valeurs (IBU)")
# plt.savefig("./predictionIBU.png")
# plt.show()

##### PREDICTION NON LINEAIRE POLYNOMIAL #####    
# X = df["BoilTime"].values.reshape(-1, 1)
# y = df["ABV"]
# degree = 2  
# poly_features = PolynomialFeatures(degree=degree)
# X_poly = poly_features.fit_transform(X.reshape(-1, 1))

# model = LinearRegression()
# model.fit(X_poly, y)

# x_pred = np.linspace(min(X), max(X), 100) 
# X_pred_poly = poly_features.transform(x_pred.reshape(-1, 1))
# y_pred = model.predict(X_pred_poly)