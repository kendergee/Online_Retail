import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
from category_encoders import OrdinalEncoder
from sklearn.decomposition import PCA
import joblib
import os
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

#篩選並分類國家的資料
# data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II.csv',encoding ='utf8')
# country_counts = data['Country'].value_counts()
# valid_countries = country_counts[country_counts >= 1000].index
# data = data[data['Country'].isin(valid_countries)]
# print(f'篩選後的資料共有 {data.shape[0]} 行')
# Country_values = data['Country'].unique()
# for country in Country_values:
#     os.makedirs(f'./{country}',exist_ok=True)
#     country_data = data[data['Country']==country]
#     country_data = country_data[country_data['Quantity'] >= 0]
#     country_data.to_csv(f'./{country}/{country}_data.csv', index=False)
#     print(f'{country} done')

def preprocessing(data):
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Year'] = data['InvoiceDate'].dt.year
    data['Quarter'] =data['InvoiceDate'].dt.quarter
    data = data.drop(['Invoice','Description','InvoiceDate','Customer ID'],axis=1)
    X = data.drop('Quantity',axis=1)
    y = data['Quantity']
    encoder = OrdinalEncoder()
    X['StockCode'] = encoder.fit_transform(X['StockCode'].values.reshape(-1, 1))
    X['Country'] = encoder.fit_transform(X['Country'].values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    stdX = StandardScaler()
    X_train = stdX.fit_transform(X_train)
    X_test = stdX.transform(X_test)
    stdy = StandardScaler()
    y_train = stdy.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = stdy.transform(y_test.values.reshape(-1, 1)).ravel()
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train,X_test,y_train,y_test

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

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Error')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Error')
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

# 模型初次訓練，評估每個國家適合哪個模型
# root_folder ='E:/vscode/Online_Retail/Stcok Prediction'
# country_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]

# for country in country_folders:
#     try:
#         file_path = os.path.join(root_folder,country,f'{country}_data.csv')
#         print(f'正在讀取{file_path}')
#         data = pd.read_csv(file_path)
#         result_columns = ['Params','mean CV MSE','CV std','Test MSE','Test R2']
#         X_train,X_test,y_train,y_test = preprocessing(data)
#         print('資料前處理完成')
#         RF_result = RF(X_train,X_test,y_train,y_test)
#         XGB_result = XGBoost(X_train,X_test,y_train,y_test)
#         GBR_result = GBR(X_train,X_test,y_train,y_test)
#         LGBM_result = LGBM(X_train,X_test,y_train,y_test)
#         results_dict = {
#         'Model': ['RF','XGB','GBR','LightGBM'],
#         result_columns[0]: [RF_result[0],XGB_result[0],GBR_result[0],LGBM_result[0]],
#         result_columns[1]: [RF_result[1],XGB_result[1],GBR_result[1],LGBM_result[1]],
#         result_columns[2]: [RF_result[2],XGB_result[2],GBR_result[2],LGBM_result[2]],
#         result_columns[3]: [RF_result[3],XGB_result[3],GBR_result[3],LGBM_result[3]],
#         result_columns[4]: [RF_result[4],XGB_result[4],GBR_result[4],LGBM_result[4]],
#         }

#         df_results = pd.DataFrame(results_dict)
#         df_results.to_csv(os.path.join(root_folder,country,f'{country} model_results.csv'), index=False)
#         print(f"結果已保存到 {country} model_results.csv")
#     except Exception as e:
#         print(f"處理 {country} 的資料時發生錯誤: {e}")

# 每個國家適合的模型
model_dict ={'RF':['EIRE','Finland','Germany','Italy','Spain','Sweden','United Kingdom'],
             'XGBoost':['Australia','Channel Islands','France'],
             'GBR':['Cyprus','Switzerland'],
             'LGBM':['Belgium','Netherlands','Norway','Portugal']}

# 第一次調整參數
param_grid = {
    'RF': {
        'n_estimators': [100, 200, 300],        # 樹的數量
        'max_depth': [10, 20],                  # 樹的最大深度
        'min_samples_split': [2, 5],            # 節點切分的最小樣本數
        'min_samples_leaf': [1, 2],             # 葉節點的最小樣本數
        'bootstrap': [True, False]              # 是否使用 bootstrap 抽樣
    },
    'XGBoost': {
        'learning_rate': [0.01, 0.05, 0.1],     # 學習率
        'n_estimators': [100, 200, 300],        # 樹的數量
        'max_depth': [3, 5, 7],                 # 樹的最大深度
        'subsample': [0.8, 1.0],                # 樣本抽樣比例
        'colsample_bytree': [0.8, 1.0],         # 每棵樹使用的特徵比例
        'reg_alpha': [0, 0.1],                  # L1 正則化
        'reg_lambda': [1, 1.5]                  # L2 正則化
    },
    'GBR': {
        'learning_rate': [0.01, 0.05, 0.1],     # 學習率
        'n_estimators': [100, 200, 300],        # 樹的數量
        'max_depth': [3, 5],                    # 樹的最大深度
        'min_samples_split': [2, 5],            # 節點切分的最小樣本數
        'min_samples_leaf': [1, 2],             # 葉節點的最小樣本數
        'subsample': [0.8, 1.0]                 # 樣本抽樣比例
    },
    'LGBM': {
        'learning_rate': [0.01, 0.05, 0.1],     # 學習率
        'n_estimators': [100, 200, 300],        # 樹的數量
        'max_depth': [5, 7],                    # 樹的最大深度
        'num_leaves': [20, 31],                 # 每棵樹的葉子節點數
        'subsample': [0.8, 1.0],                # 樣本抽樣比例
        'colsample_bytree': [0.8, 1.0],         # 每棵樹使用的特徵比例
        'reg_alpha': [0, 0.1],                  # L1 正則化
        'reg_lambda': [1, 1.5]                  # L2 正則化
    }
}

model_mapping = {
    'RF': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'GBR': GradientBoostingRegressor(),
    'LGBM': LGBMRegressor()
}

def adaptive_model_tuning(model_name, X_train, y_train):
    model = model_mapping[model_name]
    param_grid_for_model = param_grid[model_name]
    search = RandomizedSearchCV(model, param_grid_for_model, n_iter=30, cv=5, scoring='neg_mean_squared_error', random_state=42,n_jobs=8)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_model = search.best_estimator_
    
    return  best_params, best_model

for model_name, countries in model_dict.items():
    for country in countries:
        print(f"處理 {country} 的 {model_name} 模型")
        
        # 讀取資料
        file_path = f'./{country}/{country}_data.csv'
        data = pd.read_csv(file_path)

        X_train, X_test, y_train, y_test = preprocessing(data)
        
        # 初步調參
        best_params, best_model = adaptive_model_tuning(model_name, X_train, y_train)
        plot_learning_curve(best_model, X_train, y_train)
        print(f"{country} 的最佳參數: {best_params}")
        
        # 測試結果
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{country} 調參後的 MSE: {mse}, R²: {r2}")
        
        # 保存結果到 CSV
        result_df = pd.DataFrame([{'Model': model_name, 'Best Params': best_params, 'MSE': mse, 'R²': r2}])
        result_df.to_csv(f'./{country}/{country}_tuned_results.csv', index=False)
        print(f"{country} 的結果已保存\n")