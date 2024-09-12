import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
import os

# 取得當前腳本的路徑，並設置為工作目錄
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 確認工作目錄是否正確更改
print(f"Current working directory: {os.getcwd()}")

Origin = pd.read_csv('/Users/kendergee/Desktop/vscode/online_retail/online_retail_II.csv')
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

sse = []

for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
plt.savefig('Cluster.png')

kmeans = KMeans(n_clusters=7)
kmeans.fit(X)
customer_summary['Cluster'] = kmeans.labels_
customer_summary.to_csv('customer_summary.csv', index=False)
print('客戶分群完成')

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

plt.figure()  # 創建新的圖形
sns.scatterplot(data=customer_summary, x='Price', y='Frequency', hue='Cluster', palette='Set1')
plt.title('Customer Scatter Plot')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('Customer_Scatter.png')  # 儲存當前圖像
plt.show()

# 第二個圖表：價格箱型圖
plt.figure()  # 創建新的圖形
sns.boxplot(data=customer_summary, x='Cluster', y='Price')
plt.title('Price Distribution')
plt.xlabel('Cluster')
plt.ylabel('Price')
plt.savefig('Price_Distribution.png')  # 儲存當前圖像
plt.show()

# 第三個圖表：購買頻率箱型圖
plt.figure()  # 創建新的圖形
sns.boxplot(data=customer_summary, x='Cluster', y='Frequency')
plt.title('Frequency Distribution')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.savefig('Frequency_Distribution.png')  # 儲存當前圖像
plt.show()

Origin_merged = Origin.merge(customer_summary[['Customer ID', 'Cluster']], on='Customer ID',how='left')
Origin_merged.to_csv('online_retail_II_clustered.csv', index=False)

for i in range(7):
    print(f'Cluster {i}')
    print(len(customer_summary[customer_summary['Cluster'] == i]))

cluster_spending = customer_summary.groupby('Cluster')['Price'].sum()

# 2. 計算總消費金額
total_spending = cluster_spending.sum()

# 3. 計算每個群組消費金額的百分比
cluster_spending_percentage = (cluster_spending / total_spending) * 100

# 4. 計算每個群組的顧客數量
cluster_customer_count = customer_summary.groupby('Cluster')['Customer ID'].nunique()

# 5. 創建標籤，包含群組名稱、百分比和顧客數量
labels = [f'Cluster {i}\n{cluster_customer_count[i]} customers' for i in cluster_spending_percentage.index]

# 6. 繪製圓餅圖
fig, ax = plt.subplots()
ax.pie(cluster_spending_percentage, labels=labels, autopct='%1.1f%%', startangle=90)

# 讓圓餅圖成為圓形
ax.axis('equal')

# 添加標題
plt.title('Cluster Spending Percentage with Customer Count')

# 儲存並顯示圖表
plt.savefig('Cluster_Spending_Pie_Chart.png')
plt.show()