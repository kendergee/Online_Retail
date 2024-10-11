import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
from category_encoders import OrdinalEncoder
import joblib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import pymannkendall as mk


data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II.csv')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Year'] = data['InvoiceDate'].dt.year
data['Quater'] =data['InvoiceDate'].dt.quarter
data = data.drop(['Invoice','Description','InvoiceDate','Customer ID'],axis=1)

# 視覺化：quarter變化與quantity的變化幅度
# data.set_index('InvoiceDate', inplace=True)
# data.resample('Q')['Quantity'].sum().plot(kind='line', title='Quarterly Quantity Trends')
# plt.xlabel('Quarter')
# plt.ylabel('Total Quantity')
# plt.show()

# acf, pacf檢定
# print('start')
# plot_acf(data['Quantity'])
# plt.show()
# plot_pacf(data['Quantity'])
# plt.show()
# print('end')

# mannkendall檢定
# sample_df = data[['InvoiceDate', 'Quantity']].sample(frac=0.05, random_state=42)
# sample_df = sample_df.sort_values('InvoiceDate')
# result = mk.original_test(sample_df['Quantity'])
# print(result)

# PCA分析
# encoder = OrdinalEncoder()
# X = data.drop(['Quantity'],axis=1)
# X['StockCode'] = encoder.fit_transform(X['StockCode'].values.reshape(-1, 1))
# X['Country'] = encoder.fit_transform(X['Country'].values.reshape(-1, 1))
# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)
# pca = PCA()
# pca.fit(X_scaled)
# # 計算累積解釋變異率
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# # 繪製 Elbow 曲線
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Elbow Method for PCA')
# plt.grid(True)
# plt.show()

X = data.drop('Quantity',axis=1)
y = data['Quantity']
encoder = OrdinalEncoder()
X['StockCode'] = encoder.fit_transform(X['StockCode'].values.reshape(-1, 1))
X['Country'] = encoder.fit_transform(X['Country'].values.reshape(-1, 1))
# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化數據
stdX = StandardScaler()
X_train = stdX.fit_transform(X_train)
X_test = stdX.transform(X_test)

# 標準化目標變數
stdy = StandardScaler()
y_train = stdy.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test = stdy.transform(y_test.values.reshape(-1, 1)).ravel()

result_columns = ['Params','mean CV MSE','CV std','Test MSE','Test R2']
print('資料前處理完成')

def RF(X_train,X_test,y_train,y_test):
    print('RF開始')
    RF = RandomForestRegressor()
    cv_scores = cross_val_score(RF,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mean_cv_mse = -np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f'Mean CV MSE: {mean_cv_mse}')
    print(f'Std CV: {std_cv}')
    RF.fit(X_train,y_train)
    y_pred = RF.predict(X_test)

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('RF結束')

    RF_result = [RF.get_params(),mean_cv_mse,std_cv,mse,r2]

    return RF_result

def XGBoost(X_train,X_test,y_train,y_test):
    print('XGB開始')
    XGB = XGBRegressor()
    cv_scores = cross_val_score(XGB,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mean_cv_mse = -np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f'Mean CV MSE: {mean_cv_mse}')
    print(f'Std CV: {std_cv}')
    XGB.fit(X_train,y_train)
    y_pred = XGB.predict(X_test)

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('XGB結束')

    XGB_result = [XGB.get_params(),mean_cv_mse,std_cv,mse,r2]

    return XGB_result


def GBR(X_train,X_test,y_train,y_test):
    print('GBR開始')
    GBR = GradientBoostingRegressor()
    cv_scores = cross_val_score(GBR,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mean_cv_mse = -np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f'Mean CV MSE: {mean_cv_mse}')
    print(f'Std CV: {std_cv}')
    GBR.fit(X_train,y_train)
    y_pred = GBR.predict(X_test)

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('GBR結束')

    GBR_result = [GBR.get_params(),mean_cv_mse,std_cv,mse,r2]

    return GBR_result

def LGBM(X_train,X_test,y_train,y_test):
    print('LGBM開始')
    LGBM = LGBMRegressor()
    cv_scores = cross_val_score(LGBM,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mean_cv_mse = -np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f'Mean CV MSE: {mean_cv_mse}')
    print(f'Std CV: {std_cv}')
    LGBM.fit(X_train,y_train)
    y_pred = LGBM.predict(X_test)

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('LGBM結束')

    LGBM_result = [LGBM.get_params(),mean_cv_mse,std_cv,mse,r2]

    return LGBM_result

RF_result = RF(X_train,X_test,y_train,y_test)
XGB_result = XGBoost(X_train,X_test,y_train,y_test)
GBR_result = GBR(X_train,X_test,y_train,y_test)
LGBM_result = LGBM(X_train,X_test,y_train,y_test)

results_dict = {
    'Model': ['RF','XGB','GBR','LightGBM'],
    result_columns[0]: [RF_result[0],XGB_result[0],GBR_result[0],LGBM_result[0]],
    result_columns[1]: [RF_result[1],XGB_result[1],GBR_result[1],LGBM_result[1]],
    result_columns[2]: [RF_result[2],XGB_result[2],GBR_result[2],LGBM_result[2]],
    result_columns[3]: [RF_result[3],XGB_result[3],GBR_result[3],LGBM_result[3]],
    result_columns[4]: [RF_result[4],XGB_result[4],GBR_result[4],LGBM_result[4]],
}

df_results = pd.DataFrame(results_dict)
df_results.to_csv('model_results.csv', index=False)
print("結果已保存到 model_results.csv")