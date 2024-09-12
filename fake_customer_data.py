import pandas as pd
import os
import numpy as np

scipt_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(scipt_dir)

datasets = pd.read_csv('customer_classify/quantile/customer_summary.csv')
datasets['Average spending'] = datasets['Price'] / datasets['Frequency']
datasets['Average spending'] = datasets['Average spending'].round(2)
datasets['Price'] = datasets['Price'].round(2)
datasets['Total spending'] = datasets['Price']
datasets = datasets.drop(columns=['Price'])

np.random.seed(0)
ages = np.random.randint(18, 80, datasets.shape[0])
genders = np.random.choice(['Male','Female','Other'], datasets.shape[0])
incomes = np.random.randint(10000, 100000, datasets.shape[0])
education = np.random.choice(['High School','College','Bachelor','Master','PhD'], datasets.shape[0])
mariage = np.random.choice(['Single','Married'], datasets.shape[0])
children = np.random.randint(0, 5, datasets.shape[0])

datasets['Age'] = ages
datasets['Gender'] = genders
datasets['Income'] = incomes
datasets['Education'] = education
datasets['Mariage'] = mariage
datasets['Children'] = children

new_columns = ['Customer ID', 'Age','Gender','Income','Education','Country','Mariage','Children','Total spending','Frequency','Average spending','Quartile']
datasets = datasets.reindex(columns=new_columns)

datasets.to_csv('fake_customer_data.csv', index=False)
print('fake_customer_data.csv has been created')