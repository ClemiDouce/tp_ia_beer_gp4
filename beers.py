import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./recipeData.csv", encoding="latin1")
orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)
ser = df.isna().mean() * 100

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
print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)
columns_to_bin = ["Size(L)", "OG", "FG", "Color", "BoilSize"]
for column in columns_to_bin:
    df[f"{column}_Binned"] = pd.qcut(df[column], q=5, labels=False, duplicates="drop")

##### SCATTER #####
fig, ax = plt.subplots()
cm_bright = ListedColormap(["#FF4500", "#0000CC"])
ax.scatter(df[df.columns[4]], df[df.columns[8]], cmap=cm_bright, edgecolors="k")
fig.savefig("./rapport/ABV-BoilTime.png")

print(df)

plt.matshow(df.corr())
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()