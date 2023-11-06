import pandas as pd

df = pd.read_csv("bank_marketing.csv")

df.head()

df.isnull().sum()

df.dtypes

df.columns

df.dropna(inplace=True)
df.drop_duplicates(inplace= True)
# -----------------------------------

def binarycols(df, col):
  df[col].replace({'yes': 1, 'no': 0}, inplace=True)

def multiclasscols(df, col):
  df = pd.get_dummies(df, columns=[col], prefix=[col])
  return df

# --------------------------------------

bincols = ['default', "housing", "loan", "deposit"]
classcols = ["job", "marital", "education", "contact", "month", "poutcome"]

for i in bincols:
  binarycols(df, i)

for i in classcols:
  df = multiclasscols(df, i)

df.head()

# ------------------------------

x = df.drop(columns = 'deposit')
y = df[['deposit']]

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape

# ------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

mod = KNeighborsClassifier(n_neighbors= 11)
mod.fit(xtrain, ytrain)

# -----------------------------

ypred = mod.predict(xtest)

accuracy_score(ypred, ytest)

classification_report(ytest, ypred)