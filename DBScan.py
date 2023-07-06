from sklearn import datasets
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

df = pd.read_csv("steam-200k.csv", header=None, index_col=None, names=['UserID', 'Game', 'Action', 'Hours', 'Other'])
df.head()
df_purchased_games = df.loc[df['Action'] == 'purchase']
df_played_games = df.loc[df['Action'] == 'play']

#here we compute the number of games a user has played
user_counts = df_played_games.groupby('UserID')['UserID'].agg('count').sort_values(ascending=False)
#here we compute the number of hours he has played
hours_played = df_played_games.groupby('UserID')['Hours'].agg(np.sum).sort_values(ascending=False)

#df creation
user_df_played_games = pd.DataFrame({'UserID': user_counts.index, 'nb_played_games': user_counts.values})
user_df_hours_played = pd.DataFrame({'UserID': hours_played.index, 'hours_played': hours_played.values})

#merge to have one entry per user with number of hours played and number of played games
data = pd.merge(user_df_played_games, user_df_hours_played, on='UserID')

# AGNES
del data['UserID'] #don't need this for AGNES
train_data = data.to_numpy()

clustering = DBSCAN(eps=100, min_samples=10).fit(train_data)
plt.figure(figsize=(12, 6))

d0 = data[clustering.labels_ == 0]
plt.scatter (d0 ['nb_played_games'], d0 ['hours_played'], c="red", marker="o", label="label0")
d1 = data[clustering.labels_ == 1]
plt.scatter(d1['nb_played_games'], d1['hours_played'], c="green", marker="*", label="label1")
d2 = data[clustering.labels_ == 2]
plt.scatter(d2['nb_played_games'], d2['hours_played'], c="blue", marker="+", label="label2")
plt.xlabel("nb_played_games")
plt.ylabel("hours_played")
plt.title("DBScan clustering")
plt.show()

confusion_matrix()