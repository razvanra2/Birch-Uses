from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

data = pd.read_csv('minute_weather.csv')
sampled_df = data[(data['rowID'] % 10) == 0]
print(sampled_df.describe())

del sampled_df['rain_accumulation']
del sampled_df['rain_duration']

rows_before = sampled_df.shape[0]
sampled_df = sampled_df.dropna()
rows_after = sampled_df.shape[0]


features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
        'max_wind_speed','relative_humidity']
select_df = sampled_df[features]
X = StandardScaler().fit_transform(select_df)


birch = Birch(n_clusters=12)
birch.fit(X)

sub_centers = birch.subcluster_centers_
kmeans = KMeans(n_clusters=12)
kmeans.fit(sub_centers)
centers = kmeans.cluster_centers_
# Function that creates a DataFrame with a column for Cluster Number

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	Z = [np.append(A, index) for index, A in enumerate(centers)]

	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

# Function that creates Parallel Plots
def parallel_plot(data):
        my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
        plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
        parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
        plt.show()

P = pd_centers(features, centers)

#dry days
parallel_plot(P[P['relative_humidity'] < -0.5])

# warm days
parallel_plot(P[P['air_temp'] > 0.5])

#cool days
parallel_plot(P[(P['relative_humidity'] > 0.5) & (P['air_temp'] < 0.5)])
