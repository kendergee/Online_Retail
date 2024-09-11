import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns

Origin = pd.read_csv('online_retail_II.csv')
Origin = Origin.dropna()
datasets = Origin.drop(['StockCode', 'Description','Quantity','InvoiceDate'], axis=1)

customer_summary = datasets.groupby('Customer ID').agg({
    'Price': 'sum',
    'Country': 'first',
    'Invoice': 'nunique'
}).reset_index()
customer_summary = customer_summary.rename(columns={'Invoice': 'Frequency'})
customer_summary = customer_summary[customer_summary['Price']>0]
print('資料前處理完成')

X = customer_summary[['Price']]

# sse = []

# for k in range(1,11):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(X)
#     sse.append(kmeans.inertia_)

# plt.plot(range(1,11), sse)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.show()

# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)
# customer_summary['Cluster'] = kmeans.labels_
# customer_summary.to_csv('customer_summary.csv', index=False)
print('客戶分群完成')
customer_summary = pd.read_csv('customer_summary.csv')

cluster_summary = customer_summary.groupby('Cluster').agg({
    'Customer ID': 'count',
    'Price': 'mean',
    'Frequency': 'mean',
    'Country': lambda x: x.value_counts().index[0]
}).reset_index()
cluster_summary = cluster_summary.rename(columns={
    'Customer ID': 'Count',
    'Price': 'Average Price',
    'Frequency': 'Average Frequency',
    'Country': 'Main Country'
})

# sns.scatterplot(data=customer_summary, x='Price', y='Frequency', hue='Cluster')
# plt.title('Customer Clustering')
# plt.xlabel('Price')
# plt.ylabel('Frequency')
# plt.show()

# sns.boxplot(data=customer_summary, x='Cluster', y='Price')
# plt.title('Price Distribution')
# plt.xlabel('Cluster')
# plt.ylabel('Price')
# plt.show()

# sns.boxplot(data=customer_summary, x='Cluster', y='Frequency')
# plt.title('Frequency Distribution')
# plt.xlabel('Cluster')
# plt.ylabel('Frequency')
# plt.show()

Origin_merged = Origin.merge(customer_summary[['Customer ID', 'Cluster']], on='Customer ID',how='left')
Origin_merged.to_csv('online_retail_II_clustered.csv', index=False)