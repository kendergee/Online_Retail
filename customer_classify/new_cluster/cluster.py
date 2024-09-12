import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

datasets = pd.read_csv('/Users/kendergee/Desktop/vscode/online_retail/fake_customer_data.csv')
datasets = datasets.dropna()
datasets = datasets[datasets['Quartile'] == 4]
datasets = datasets.drop(columns=['Quartile', 'Average spending', 'Customer ID','Children','Mariage','Country','Frequency'])

le = LabelEncoder()
for column in ['Gender', 'Education']:
    datasets[column] = le.fit_transform(datasets[column])

sc = StandardScaler()
datasets[['Total spending', 'Age', 'Income','Gender', 'Education']] = sc.fit_transform(datasets[['Total spending', 'Age', 'Income', 'Gender', 'Education']])

linked = linkage(datasets, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

for i in range(2, 20):
    clusters = fcluster(linked, i, criterion='distance')
    sil_score = silhouette_score(datasets, clusters)
    print(f'Number of clusters: {i}, Silhouette Score: {sil_score}')
