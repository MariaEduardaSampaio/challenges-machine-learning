import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering


def plot_clusters(samples, clusters):
    col_dic = {0: 'blue', 1: 'green', 2: 'orange', 3: 'cyan'}
    mrk_dic = {0: '*', 1: 'x', 2: '+', 3: '.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1],
                    color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()


df_clusters = pd.read_csv('data/clusters.csv')
print(df_clusters.head())

# Normalize the numeric features, so they are on the same scale
scaled_features = MinMaxScaler().fit_transform(df_clusters)

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
print(features_2d[0:5])

# Visualize the unclustered data points
plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_init=1, n_clusters=i)
    kmeans.fit(df_clusters)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

model = KMeans(n_clusters=4, init='k-means++', n_init=500, max_iter=1500)
km_clusters = model.fit_predict(df_clusters)
print(km_clusters)

plot_clusters(features_2d, km_clusters)

agg_model = AgglomerativeClustering(n_clusters=4)
agg_clusters = agg_model.fit_predict(df_clusters)
print(agg_clusters)

plot_clusters(features_2d, agg_clusters)
