import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

datasets = pd.read_csv('/Users/kendergee/Desktop/vscode/online_retail/fake_customer_data.csv')
datasets = datasets.dropna()
datasets = datasets[datasets['Quartile'] == 4]
datasets = datasets.drop(columns=['Quartile', 'Average spending', 'Customer ID'])

datasets['Age'] = pd.cut(datasets['Age'], bins=[0, 30, 40, 50, 60, 70, 80], labels=['0-30', '31-40', '41-50', '51-60', '61-70', '71-80'])
datasets['Income'] = pd.cut(datasets['Income'], bins=[0, 20000, 40000, 60000, 80000, 100000], labels=['0-20000', '20001-40000', '40001-60000', '60001-80000', '80001-100000'])
print(datasets.info())

le = LabelEncoder()
for column in ['Age','Gender', 'Education','Country','Mariage','Income']:
    datasets[column] = le.fit_transform(datasets[column])

sc = StandardScaler()
datasets[datasets.columns] = sc.fit_transform(datasets[datasets.columns])

pca = PCA(n_components=0.9)
datasets = pca.fit_transform(datasets)
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
