import pandas as pd

data = pd.read_csv('/Users/kendergee/Desktop/vscode/online_retail/customer_classify/new_cluster/customer_q4_with_cluster.csv')

# cluster_summary = data.groupby('Cluster').agg({
#     'Total spending': 'sum',
#     'Age': 'mean',
#     'Gender': 'mode',
#     'Education': 'mode',
#     'Country': 'mode',
#     'Mariage': 'mode',
#     'Income': 'median',
#     'Children': 'median',
#     'Frequency': 'sum',
#     'Average spending': 'mean',
#     'Customer ID': 'nunique'
# }).reset_index()

print(cluster_summary)