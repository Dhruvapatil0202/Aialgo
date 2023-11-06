# ------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------

df = pd.read_csv('uber.csv')
df.info()
# ------------------------------------------------------------

df.head()
# ------------------------------------------------------------

df.describe()
# ------------------------------------------------------------

# removal of first two columns (Unnamed: 0 and key)

df = df.drop(['Unnamed: 0', 'key'], axis = 1)
df.info()
# ------------------------------------------------------------

# Removal of Null values


df.dropna(axis = 0, inplace = True)
df.isna().sum()

# ------------------------------------------------------------

df.dtypes


# ------------------------------------------------------------

# Conversion of pickup_datetime into datetime object

df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors = 'coerce')

# ------------------------------------------------------------

# conversion of datetime into seperate columns of each time component

df= df.assign(
    second = df['pickup_datetime'].dt.second,
    minute = df.pickup_datetime.dt.minute,
    hour = df.pickup_datetime.dt.hour,
    day = df.pickup_datetime.dt.day,
    month = df.pickup_datetime.dt.month,
    year = df.pickup_datetime.dt.year,
    dayofweek = df.pickup_datetime.dt.dayofweek,
)
df = df.drop(['pickup_datetime'], axis = 1)
df.info()

# ------------------------------------------------------------


# Finding out incorrect coordinates

coordinates_to_be_dropped = df.loc[
    (df.pickup_latitude > 90) | (df.pickup_latitude < -90) |
    (df.dropoff_latitude > 90) | (df.dropoff_latitude < -90) |
    (df.pickup_longitude > 180) | (df.pickup_longitude < -180) |
    (df.dropoff_longitude > 90) | (df.dropoff_longitude < -90)
]

df.drop(coordinates_to_be_dropped, inplace = True, errors = 'ignore')


# ------------------------------------------------------------


# Implementing Function for calculating haversine distance

def dist_transform(long1, lat1, long2, lat2):
  x1, y1, x2, y2 = map(np.radians, [long1, lat1, long2, lat2])
  long_dist = x2 - x1
  lati_dist = y2 - y1
  temp = np.sin(lati_dist) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(long_dist) ** 2
  c  = 2 * np.arcsin(np.sqrt(temp)) * 6371
  return c

# ------------------------------------------------------------

df['Distance'] = dist_transform(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
    )

df.head()

# ------------------------------------------------------------


# Visualizing the data to detect outliars

plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel('Distance')
plt.ylabel('fare_amount')

# ------------------------------------------------------------

plt.figure(figsize = (18, 12))
sns.boxplot(data= df)

# ------------------------------------------------------------



# Dropping the records within specific range

df.drop(df[df['Distance'] >= 60].index, inplace = True)
df.drop(df[(df['fare_amount'] <= 0)].index, inplace = True)

df.drop(df[(df['Distance'] < 1) & (df['fare_amount'] > 100)].index, inplace = True)
df.drop(df[(df['Distance'] > 100) & (df['fare_amount'] < 100)].index, inplace = True)

plt.scatter(df['Distance'], df['fare_amount'])

# ------------------------------------------------------------


# Scaleing the values

df.dropna(axis = 0, inplace = True)

x = df['Distance'].values.reshape(-1, 1)
y = df['fare_amount'].values.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler

sd = StandardScaler()
y_std = sd.fit_transform(y)
print(y_std)

x_std = sd.fit_transform(x)
print(x_std)


# ------------------------------------------------------------

df.dropna(axis = 0, inplace = True)
df.isna().sum()

# ------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_std, y_std, test_size = 0.2)
print(x_train)

# ------------------------------------------------------------

# Training

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

print(lr.score(x_train, y_train),
lr.score(x_test, y_test))

# ------------------------------------------------------------

# Prediction

y_pred = lr.predict(x_test)
result = pd.DataFrame()
result[['Actual']] = y_test
result[['predicted']] = y_pred

result.sample(10)



# ------------------------------------------------------------


# performance metrics

from sklearn import metrics

print('Mean abs error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean abs % error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R - squared:', metrics.r2_score(y_test, y_pred))


# ------------------------------------------------------------


# Correlation check

plt.subplot(2, 2, 1)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'blue')

plt.subplot(2, 2, 2)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,lr.predict(x_train), color = 'blue')

# ------------------------------------------------------------

# Training of RandomForestREgressor

from sklearn.ensemble import RandomForestRegressor

rr = RandomForestRegressor(n_estimators= 100)
rr.fit(x_train, y_train)

# ------------------------------------------------------------


rr_y_pred = rr.predict(x_test)

rr_y_pred = np.array([[i] for i in rr_y_pred])



rrres = pd.DataFrame()
rrres[['actual']] = y_test
rrres[['predict']] = rr_y_pred

print(rrres)


# print(rr_y_pred.shape, y_test.shape)

# ------------------------------------------------------------

# performance metrics



print('Mean abs error:', metrics.mean_absolute_error(y_test, rr_y_pred))
print('Mean abs % error:', metrics.mean_absolute_percentage_error(y_test, rr_y_pred))
print('Mean squared error:', metrics.mean_squared_error(y_test, rr_y_pred))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, rr_y_pred)))
print('R - squared:', metrics.r2_score(y_test, rr_y_pred))


# ------------------------------------------------------------

plt.scatter(x_test, y_test, c = 'r', marker = '.', label = 'real')
plt.scatter(x_test, rr_y_pred, c = 'b', marker = '.', label = 'predicted')