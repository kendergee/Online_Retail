import pandas as pd

datasets = pd.read_csv('online_retail_II.csv')

# country_summary = datasets.groupby('Country').agg({
#     'Invoice': 'count',
#     'Price': 'sum',
#     'Customer ID': 'nunique'
# }).reset_index()

# print(country_summary.sort_values('Price', ascending=False).head(10))

datasets['InvoiceDate'] = pd.to_datetime(datasets['InvoiceDate'])
datasets_2009 = datasets[datasets['InvoiceDate'].dt.year == 2009]
datasets_2010 = datasets[datasets['InvoiceDate'].dt.year == 2010]
datasets_2011 = datasets[datasets['InvoiceDate'].dt.year == 2011]

print(datasets_2009['Price'].sum())