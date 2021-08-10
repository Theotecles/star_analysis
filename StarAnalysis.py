# IMPORT PACKAGES
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# IMPORT DATA
stardf = pd.read_csv("D:\KaggleData\Data\stars.csv")

# LOOK AT FIRST 5 ROWS OF DATA AND LAST 5 ROWS OF DATA
print(stardf.head())
print(stardf.tail())

# CHECK DATA TYPES
print(stardf.dtypes)

# TOTAL ROWS AND COLUMNS
print(stardf.shape)

# CHECK FOR DUPLICATE ROWS
duplicatedf = stardf[stardf.duplicated()]
print("Number of duplicate rows:", duplicatedf.shape)

# NO DUPLICATES
# CHECK FOR MISSING OR NULL VALUES
print(stardf.isnull().sum())

# CONVERT COLOR AND SPECTRAL CLASS TO NUMBERS
print(stardf['Color'].unique())
print(stardf['Spectral_Class'].unique())

color_mapping = {
    'Red': 0,
    'Blue White': 1,
    'White': 2,
    'Yellowish White': 3,
    'Blue white': 1,
    'Pale yellow orange': 4,
    'Blue': 5,
    'Blue-white': 1,
    'Whitish': 2,
    'yellow-white': 3,
    'Orange': 6,
    'White-Yellow': 3,
    'white': 2,
    'yellowish': 7,
    'Yellowish': 7,
    'Orange-Red': 8,
    'Blue-White': 1,
}

stardf.Color = [color_mapping[item] for item in stardf.Color]

class_mapping = {
    'M': 0,
    'B': 1,
    'A': 2,
    'F': 3,
    'O': 4,
    'K': 5,
    'G': 6,
}

stardf.Spectral_Class = [class_mapping[item] for item in stardf.Spectral_Class]

# TAKE OUT NON NUMERICAL DATA
stardf1 = stardf.drop(['Color'], axis=1)
stardf2 = stardf1.drop(['Spectral_Class'], axis=1)

print(stardf.shape)
print(stardf2.shape)

# SET UP SNS STANDARD AESTHETICS FOR PLOTS
sns.set()

# CREATE A BOXPLOT FOR EACH COLUMN
sns.boxplot(x=stardf2['Temperature'])
plt.show()

sns.boxplot(x=stardf2['L'])
plt.show()

sns.boxplot(x=stardf2['R'])
plt.show()

sns.boxplot(x=stardf2['A_M'])
plt.show()

sns.boxplot(x=stardf2['Type'])
plt.show()

# SET BIN ARGUMENT
n_obs = len(stardf2)

n_bins = round(np.sqrt(n_obs), 0)

print(n_bins)

# CREATE HISTOGRAMS FOR EACH COLUMN
plt.hist(stardf2['Temperature'], density=True, bins=15)
plt.show()

plt.hist(stardf2['L'], density=True, bins=15)
plt.show()

plt.hist(stardf2['R'], density=True, bins=15)
plt.show()

plt.hist(stardf2['A_M'], density=True, bins=15)
plt.show()

plt.hist(stardf2['Type'], density=True, bins=7)
plt.show()

plt.hist(stardf['Color'], density=True, bins=9)
plt.show()

plt.hist(stardf['Spectral_Class'], density=True, bins=6)
plt.show()

# CREATE A BSWARM PLOT FOR EACH COLUMN
sns.swarmplot(x = 'Type', y = 'Temperature', data = stardf2, size = 1.5)
plt.show()

sns.swarmplot(x = 'Type', y = 'L', data = stardf2, size = 1.5)
plt.show()

sns.swarmplot(x = 'Type', y = 'R', data = stardf2, size = 1.5)
plt.show()

sns.swarmplot(x = 'Type', y = 'A_M', data = stardf2, size = 1.5)
plt.show()

# ALL DATA IS HEAVILY LEFT SKEWED EXCEPT FOR A_M WHICH APPEARS TO BE BIMODAL

# CREATE A CORRELATION MATRIX
c = stardf2.corr()
print(c)

# CREATE SCATTERPLOTS FOR EACH VARIABLE AS WELL
plt.scatter(stardf2['Temperature'], stardf2['Type'])
plt.show()

plt.scatter(stardf2['L'], stardf2['Type'])
plt.show()

plt.scatter(stardf2['R'], stardf2['Type'])
plt.show()

plt.scatter(stardf2['A_M'], stardf2['Type'])
plt.show()

# FIND SUMMARY STATISTICS
print(stardf2.describe())

# CREATE 95% CONFIDENCE INTERVALS

conf_int_temp = np.percentile(stardf['Temperature'], [2.5, 97.5])

conf_int_l = np.percentile(stardf['L'], [2.5, 97.5])

conf_int_r = np.percentile(stardf['R'], [2.5, 97.5])

print(f"95% confidence interval for Temperature = {conf_int_temp}")

print(f"95% confidence interval for L = {conf_int_l}")

print(f"95% confidence interval for R = {conf_int_r}")

# SPLIT DATA INTO TRAINING AND TEST SPLITS
y = stardf['Type']
X = stardf.drop('Type', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# DETERMINE OPTIMAL K
n_range = range(1,26)
scores = {}
scores_list = []
for n in n_range:
    k = KNeighborsClassifier(n_neighbors=n)
    k.fit(X_train, y_train)
    predictions = k.predict(X_test)
    scores[n] = sum(y_test == predictions) / len(y_test)
    scores_list.append(sum(y_test == predictions) / len(y_test))

print(scores_list)
plt.plot(n_range, scores_list)
plt.show()

#CREATE THE KNN MODEL
k = KNeighborsClassifier(n_neighbors=1)
k.fit(X_train, y_train)

# PREDICT
model_predictions = k.predict(X_train)
print(sum(y_train == model_predictions) / len(y_train))

predictions = k.predict(X_test)
print(sum(y_test == predictions) / len(y_test))
