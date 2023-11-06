import pandas as pd
import seaborn as sns

df= pd.read_csv('Churn_Modelling.csv')

df.shape

df.columns

df.head()

x = df[["CreditScore", 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']


sns.countplot(x = y)

y.value_counts()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
x_scaled

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y , test_size= 0.25)
x_scaled.shape, x_train.shape, x_test.shape

from sklearn.neural_network import MLPClassifier


ann = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                    random_state = 0,
                    max_iter = 100,
                    activation = 'relu')


ann.fit(x_train, y_train)

y_pred = ann.predict(x_test)

y_test.value_counts()

from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))