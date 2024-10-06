import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import os

data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II_2010~2011.csv')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['InvoiceDate'] = data['InvoiceDate'].dt.quarter
data['SalesAmount'] = data['Quantity']*data['Price']

#全球趨勢分析，繪製每個quarter的銷售筆數和銷售額度的bar plot
Quarter_SalesAmount = data.groupby('InvoiceDate')['SalesAmount'].sum()
Quarter_InvoiceCounts = data.drop_duplicates(subset='Invoice').groupby('InvoiceDate')['Invoice'].count()

sns.barplot(x=Quarter_InvoiceCounts.index,y=Quarter_InvoiceCounts.values)
plt.title('Global Quarter Transaction Counts')
plt.xlabel('Quarter')
plt.ylabel('Transactions')
plt.savefig('2010~2011 Global Quarter Transaction Counts picture')
plt.show()

sns.barplot(x=Quarter_SalesAmount.index,y=Quarter_SalesAmount.values)
plt.title('Global Quarter SalesAmount')
plt.xlabel('Quarter')
plt.ylabel('SalesAmount')
plt.savefig('2010~2011 Global Quarter SalesAmount picture')
plt.show()

#區域分析，繪製每個國家每個quarter的銷售筆數和銷售額度的bar plot
country = data['Country'].unique()
country_grouped = data.groupby('Country')
country_df ={}
for i in country:
    country_df[i] = country_grouped.get_group(i)
    country_df[i] = country_df[i][country_df[i]['SalesAmount']>=0]
    
    Quarter_SalesAmount = country_df[i].groupby('InvoiceDate')['SalesAmount'].sum()
    Quarter_InvoiceCounts = country_df[i].drop_duplicates(subset='Invoice').groupby('InvoiceDate')['Invoice'].count()

    folder_path = os.path.join(r'E:\vscode\Online_Retail\Quarter trend 2010~2011\Country png',f'2010~2011 {i}')
    os.mkdir(folder_path)

    sns.barplot(x=Quarter_InvoiceCounts.index,y=Quarter_InvoiceCounts.values)
    plt.title(f'{i} Quarter Transaction Counts')
    plt.xlabel('Quarter')
    plt.ylabel('Transactions')
    img = f'2010~2011 {i} Quarter Transaction Counts.png'
    savepath = os.path.join(folder_path,img)
    plt.savefig(savepath,dpi=300)
    plt.clf()

    sns.barplot(x=Quarter_SalesAmount.index,y=Quarter_SalesAmount.values)
    plt.title(f'{i} Quarter SalesAmount')
    plt.xlabel('Quarter')
    plt.ylabel('SalesAmount')
    img = f'2010~2011 {i} Quarter SalesAmount.png'
    savepath = os.path.join(folder_path,img)
    plt.savefig(savepath,dpi=300)
    plt.clf()

