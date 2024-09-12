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

customer_summary['Quartile'] = pd.qcut(customer_summary['Price'], q=4, labels=False) + 1

x = customer_summary['Price']
sns.boxplot(x=x)
plt.xlabel('Whole Spending')
plt.title('Boxplot of Customer Frequency')
#plt.show()
plt.close()

quartile_summary = customer_summary.groupby('Quartile').agg({
    'Customer ID': 'count',
    'Price': 'sum'
}).reset_index().rename(columns={'Price': 'Total Spending'})

total_spending = quartile_summary['Total Spending'].sum()
quartile_summary['Percentage'] = quartile_summary['Total Spending'] / total_spending *100
print(quartile_summary)

import matplotlib.pyplot as plt

# 繪製圓餅圖
labels = quartile_summary['Quartile']
sizes = quartile_summary['Percentage']
explode = (0, 0, 0, 0.1)  # 將第一個分位數突出顯示

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # 確保圓餅圖是圓形的
plt.title('Percentage of Total Spending by Quartile')
plt.savefig('Quartile.png')
plt.show()

# 將合併後的資料保存到 CSV 文件
customer_summary.to_csv('customer_summary.csv', index=False)