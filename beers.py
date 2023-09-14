import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)
ser = df.isna().mean() * 100
# print(df)


# aze = df["Size(L)"].quantile(0.95)
df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[df["IBU"] <= 150]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

df['Bitterness_PA'] = df['IBU'] / df['ABV']  # Amertume par unité d'alcool 
df['Total_Density'] = df['OG'] - df['FG'] # Concentration de sucre dans la bière
df['OG_FG_Ratio'] = df['OG'] / df['FG'] # Taux de fermentation d'une bière


# hist = df.hist(bins=50, log=True)
# print(df.describe(include='all'))

new_len = len(df)
# print(new_len / orig_leng)
one_hot = pd.get_dummies(df["BrewMethod"])
print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)
columns_to_bin = ["Size(L)", "OG", "FG", "ABV", "IBU", "Color", "BoilSize"]
for column in columns_to_bin:
    df[f"{column}_Binned"] = pd.qcut(df[column], q=5, labels=False, duplicates="drop")


#columns_to_bin = ["Size(L)", "OG", "FG", "ABV", "IBU", "Color", "BoilSize"]
#bin_intervals = {
#    "Size(L)": [0, 10, 20, 30],
#    "OG": [1.000, 1.040, 1.060, 1.070],
#    "FG": [1.000, 1.010, 1.020, 1.030],
#    "ABV": [0, 5, 7, 10],
#    "IBU": [0, 30, 60, 90, 120],
#    "Color": [0, 10, 20, 30, 40],
#    "BoilSize": [0, 10, 20, 30, 40]
#}

#for column in columns_to_bin:
#    df[f"{column}_Binned"] = pd.cut(df[column], bins=bin_intervals[column], labels=False)

# df = df.drop(columns=columns_to_bin)



print(df)

plt.matshow(df.corr())
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()