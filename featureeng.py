import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from numpy import log, log1p
from scipy.stats import boxcox
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from pandas import get_dummies
sns.set()

df = pd.read_csv("Ames_Housing_Data.tsv", sep='\t')
df['Gr Liv Area'].hist()
#plt.show()

df = df.loc[df['Gr Liv Area'] <= 4000, :]  # reducing dataset to remove outliers
data = df.copy()
df.drop(['PID', 'Order'], axis=1, inplace=True)
numcols = df.select_dtypes('number').columns

# feature transormation
skew = 0.75
skewvals = df[numcols].skew()
skewcols = skewvals[abs(skewvals) > skew].sort_values(ascending=False)  # skewed columns greater than skew limit
field = 'SalePrice'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
df[field].hist(ax=ax1)
df[field].apply(np.log1p).hist(ax=ax2)
ax1.set(title='Data before log1p transformation', xlabel='value', ylabel='frequency')
ax2.set(title='Data after log1p transformation', xlabel='value', ylabel='frequency')
fig.suptitle('SalePrice')
#plt.show()

# transforming the remaining columns using log1p
for col in skewcols.index.values:
    if col == 'SalePrice':
        continue
    df[col].apply(np.log1p)

# removing null vals
data.isnull().sum().sort_values(ascending=False)

# selecting features relative to sale price
cleaneddf = df.loc[:, ['Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Gr Liv Area',
                        'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 'Garage Cars', 'SalePrice']]

cleaneddf.describe().T
cleaneddf.info()  # checking for null counts, Garage Cars has 1 null
cleaneddf = cleaneddf.fillna(cleaneddf['Garage Cars'].median())  # filling garage car null with median

sns.pairplot(cleaneddf, plot_kws=dict(alpha=0.1, edgecolor='none'))  # pairplot for comparing different metrics
#plt.show()

target = cleaneddf['SalePrice']
features = cleaneddf.drop(['SalePrice'], axis=1)

# polynomial features
x = features.copy()
x['0Q2'] = x['Overall Qual'] ** 2   # since overall qual has a quadratic trend, create a new col containing overall qual ** 2
x['GLA2'] = x['Gr Liv Area'] ** 2

# feature interations due to propagation of polynomial features
x['OQxYB'] = x['Overall Qual'] * x['Year Built']
x['OQ_/_LA'] = x['Overall Qual'] / x['Lot Area']  # quality per square foot

# Non-numerical variables
variablecols = df.select_dtypes('object').columns
variablecols
df['House Style'].value_counts()
pd.get_dummies(df['House Style'], drop_first=True).head()  # assigns numerical values to house style but reating new colums and assigning binary values

nbhcounts = df['Neighborhood'].value_counts()
lowcounts = list(nbhcounts[nbhcounts <= 8].index)  # grouping lowcounts

x['Neighborhood'] = df['Neighborhood'].replace(lowcounts, 'Other')
x.Neighborhood.value_counts()

def add_deviation_feature(data, feature, category):
    '''
    data: input data set
    feature: comparison variable e.g Overall Quality
    category: input variable e.g Neighborhood
    return: numerical value of feature relative to category
    This allows for comparison of Overall Quality per Neighborhood, we can then
    take the mean and std to check quality of a house compared with others in
    neighborhood
    '''

    category_qb = data.groupby(category)[feature]

    category_mean = category_qb.transform(lambda x: x.mean())
    category_std = category_qb.transform(lambda x: x.std())

    deviation_feature = (data[feature] - category_mean)/category_std
    data[feature + '_Dev_' + category] = deviation_feature


x['House Style'] = df['House Style']
add_deviation_feature(x, 'Year Built', 'House Style')
add_deviation_feature(x, 'Overall Qual', 'Neighborhood')

# Better way to deal with polynomial features
pf = PolynomialFeatures(degree=2)
features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])
feature_array = pf.transform(df[features])
df2 = pd.DataFrame(feature_array, columns=pf.get_feature_names(input_features=features))