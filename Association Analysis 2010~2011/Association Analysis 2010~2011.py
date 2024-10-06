import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

# 轉csv
# datap1 = pd.read_excel(r'E:\vscode\Online_Retail\online_retail_II_2009~2010.csv',sheet_name=0)
# datap2 = pd.read_excel(r'E:\vscode\Online_Retail\online_retail_II_2009~2010.csv',sheet_name=1)
# datap1.to_csv('online_retail_II_2009~2010.csv')
# datap2.to_csv('online_retail_II_2010~2011.csv')

# 建立商品清單2009~2010
# data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II_2010~2011.csv')
# product_table={}
# for index,row in data.iterrows():
#     stock_code = row['StockCode']
#     description = row['Description']
#     product_table[stock_code] = description
# product_table = pd.DataFrame(list(product_table.items()),columns=['StockCode','Description'])
# product_table.to_csv('Stock and Description 2010~2011.csv')

# 關聯分析
# data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II_2010~2011.csv')
# data = data.sort_values('Invoice')
# data = data[['Invoice','StockCode']]
# data = data.dropna()
# basket = data.groupby(['Invoice','StockCode']).size().unstack(fill_value=0)
# basket = basket.applymap(lambda x:1 if x>0 else 0)
# frequent_itemsets = apriori(basket,min_support=0.01,use_colnames=True)
# rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
# rules.to_csv('Association Analysis 2010~2011.csv')

#查看前十名常見的組合
data = pd.read_csv(r'E:\vscode\Online_Retail\Association Analysis 2010~2011 copy\Association Analysis 2010~2011.csv')
print(len(data))
confidence_thresh =0.5
lift_thresh= 1.5
support = 0.03
data = data[(data['confidence']>=confidence_thresh) & (data['lift']>=lift_thresh) & (data['support']>=support)]
data = data.sort_values(by='lift',ascending=False)
print(data)