import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

data = pd.read_csv('/Users/kendergee/Desktop/vscode/online_retail/fake_customer_data.csv')
data = data.dropna()
data = data[data['Quartile'] == 4]
datasets = data.drop(columns=['Quartile', 'Average spending', 'Customer ID'])

datasets['Age'] = pd.cut(datasets['Age'], bins=[0, 30, 40, 50, 60, 70, 80], labels=['0-30', '31-40', '41-50', '51-60', '61-70', '71-80'])
datasets['Income'] = pd.cut(datasets['Income'], bins=[0, 20000, 40000, 60000, 80000, 100000], labels=['0-20000', '20001-40000', '40001-60000', '60001-80000', '80001-100000'])

le = LabelEncoder()
le_labelmap ={}
for column in ['Age','Gender', 'Education','Country','Mariage','Income']:
    datasets[column] = le.fit_transform(datasets[column])
    datasets[column] = datasets[column].astype('int64')
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    le_labelmap[column] = label_mapping

sc = StandardScaler()
datasets[datasets.columns] = sc.fit_transform(datasets[datasets.columns])

linked = linkage(datasets, method='ward')

# plt.figure(figsize=(10, 7),dpi=1000)
# dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
# plt.title('Dendrogram')
# plt.xlabel('Sample index')
# plt.ylabel('Distance')
# plt.savefig('dendrogram.png')
# plt.show()

model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
datasets['Cluster'] = model.fit_predict(datasets)


le_column = ['Age','Gender', 'Education','Country','Mariage','Income']

datasets[['Age','Gender','Income','Education','Country','Mariage','Children','Total spending','Frequency']] = sc.inverse_transform(datasets[['Age','Gender','Income','Education','Country','Mariage','Children','Total spending','Frequency']])
datasets[le_column] = datasets[le_column].astype(int)

data['Cluster'] = datasets['Cluster']

# for col in le_column:
#     print(col)
#     print(datasets[col].unique())

# for col, label_mapping in le_labelmap.items():
#     print(col)
#     print(label_mapping)

# for col in datasets.columns:
#     def map_label(x):
#         try:
#             return label_mapping[x]
#         except KeyError:
#             print(f"KeyError: {x} not found in {col}")
#             return None
    
#     datasets[col] = datasets[col].apply(map_label)

# for col in datasets.columns:
#     print(f"Processing column: {col}")
#     print(f"Unique values in column: {datasets[col].unique()}")
#     print(f"Mapping for column: {label_mapping}")

# for col in le_column:
#     label_mapping = le_labelmap[col]
#     datasets[col] = datasets[col].astype(str)
#     datasets[col] = datasets[col].apply(lambda x: label_mapping.get(x, None)) 

# datasets = pd.DataFrame(datasets.reset_index(drop=True, inplace=True))
# data = pd.DataFrame(data.reset_index(drop=True, inplace=True))

# print(datasets.head())
# print(data.head())
data.to_csv('customer_q4_with_cluster.csv', index=False)
print('customer_data_with_cluster.csv has been created')