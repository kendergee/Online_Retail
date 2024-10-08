import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
from category_encoders import OrdinalEncoder

data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II.csv')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Year'] = data['InvoiceDate'].dt.year
data['Quater'] =data['InvoiceDate'].dt.quarter
data = data.drop(['Invoice','Description','InvoiceDate','Customer ID'],axis=1)

X = data.drop('Quantity',axis=1)
y =data['Quantity']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

encoder1 = OrdinalEncoder()
X_train['StockCode'] = encoder1.fit_transform(X_train['StockCode'])
X_test['StockCode'] = encoder1.transform(X_test['StockCode'])
encoder2 = OrdinalEncoder()
X_train['Country'] = encoder2.fit_transform(X_train['Country'])
X_test['Country'] = encoder2.transform(X_test['Country'])

stdX = StandardScaler()
X_train = stdX.fit_transform(X_train)
X_test = stdX.transform(X_test)

stdy = StandardScaler()
y_train = stdy.fit_transform(y_train.values.reshape(-1, 1))
y_test = stdy.transform(y_test.values.reshape(-1, 1))

y_train = y_train.ravel()
y_test = y_test.ravel()

result_columns = ['Best Params','Best CV score','Test MSE','Test R2']
print('資料前處理完成')

def RF(X_train,X_test,y_train,y_test):
    print('RF開始')
    RF = RandomForestRegressor()
    param_grid ={
        'n_estimators':[50,100,200],
        'max_depth':[None,10,20,30],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[1,2,4],
        'max_features':['sqrt','log2']
    }
    grid_search = GridSearchCV(estimator=RF,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs = -1)
    grid_search.fit(X_train,y_train)
    print('RandomForesRegressor')
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (Negative MSE):", grid_search.best_score_)
    print('   ')

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = stdy.inverse_transform(y_pred.reshape(-1, 1))  # 將預測結果轉換回原尺度

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('RF結束')

def XGBoost(X_train,X_test,y_train,y_test):
    print('XGB開始')
    XGB = XGBRegressor()
    param_grid = {
    'n_estimators': [50, 100, 200],        # 樹的數量
    'learning_rate': [0.01, 0.1, 0.2],     # 學習率（步長）
    'max_depth': [3, 5, 7],                # 樹的最大深度
    'subsample': [0.6, 0.8, 1.0],          # 每棵樹樣本的取樣比例
    'colsample_bytree': [0.6, 0.8, 1.0],   # 每棵樹使用特徵的取樣比例
    'gamma': [0, 0.1, 0.2],                # 損失函數最小化所需的劃分點減益
    'reg_alpha': [0, 0.01, 0.1],           # L1 正則化項的權重（防止過擬合）
    'reg_lambda': [1, 0.1, 0.5],            # L2 正則化項的權重（防止過擬合）
    'device':['cuda'],
    'tree_method':['hist']
    }
    grid_search = GridSearchCV(estimator= XGB,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs = -1)
    grid_search.fit(X_train,y_train)
    print('XGBoost')
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (Negative MSE):", grid_search.best_score_)
    print('   ')

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = stdy.inverse_transform(y_pred.reshape(-1, 1))  # 將預測結果轉換回原尺度

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('XGB結束')

    XGB_result = [grid_search.best_estimator_,grid_search.best_score_,mse,r2]

    return XGB_result


def GBR(X_train,X_test,y_train,y_test):
    print('GBR開始')
    GBR = GradientBoostingRegressor()
    param_grid = {
    'n_estimators': [50, 100, 200],        # 樹的數量
    'learning_rate': [0.01, 0.05, 0.1],    # 學習率，控制每棵樹對結果的影響
    'max_depth': [3, 5, 7],                # 每棵樹的最大深度，控制模型複雜度
    'min_samples_split': [2, 5, 10],       # 分裂節點所需的最小樣本數
    'min_samples_leaf': [1, 2, 4],         # 每個葉子節點的最小樣本數
    'subsample': [0.6, 0.8, 1.0],          # 每棵樹樣本的取樣比例
    'max_features': ['sqrt', 'log2']  # 每次分裂時考慮的最大特徵數
    }

    grid_search = GridSearchCV(estimator= GBR,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs = -1)
    grid_search.fit(X_train,y_train)
    print('GradientBoostRegressor')
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (Negative MSE):", grid_search.best_score_)
    print('   ')

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = stdy.inverse_transform(y_pred.reshape(-1, 1))  # 將預測結果轉換回原尺度

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('GBR結束')

def LGBM(X_train,X_test,y_train,y_test):
    print('LGBM開始')
    LGBM = LGBMRegressor()
    param_grid = {
    'n_estimators': [50, 100, 200],        # 樹的數量
    'learning_rate': [0.01, 0.05, 0.1],    # 學習率，控制每棵樹對結果的影響
    'max_depth': [3, 5, 7, -1],            # 每棵樹的最大深度，-1 表示不設置深度限制
    'num_leaves': [20, 31, 40],            # 每棵樹的葉子節點數量，控制模型複雜度
    'min_child_samples': [5, 10, 20],      # 每個葉子節點的最小樣本數，防止過擬合
    'subsample': [0.6, 0.8, 1.0],          # 每棵樹樣本的取樣比例
    'colsample_bytree': [0.6, 0.8, 1.0],   # 每棵樹使用特徵的取樣比例
    'reg_alpha': [0, 0.01, 0.1],           # L1 正則化項的權重（防止過擬合）
    'reg_lambda': [0, 0.01, 0.1],           # L2 正則化項的權重（防止過擬合）
    'device':['gpu']
    }

    grid_search = GridSearchCV(estimator= LGBM,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs = -1)
    grid_search.fit(X_train,y_train)
    print('LightGradientBoostRegressor')
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score (Negative MSE):", grid_search.best_score_)
    print('    ')

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred = stdy.inverse_transform(y_pred.reshape(-1, 1))  # 將預測結果轉換回原尺度

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('LGBM結束')

    LGBM_result = [grid_search.best_estimator_,grid_search.best_score_,mse,r2]

    return LGBM_result

# RF(X_train,X_test,y_train,y_test)
# XGB_result = XGBoost(X_train,X_test,y_train,y_test)
# GBR(X_train,X_test,y_train,y_test)
LGBM_result = LGBM(X_train,X_test,y_train,y_test)

results_dict = {
    'Model': ['LightGBM'],
    result_columns[0]: [LGBM_result[0]],
    result_columns[1]: [LGBM_result[1]],
    result_columns[2]: [LGBM_result[2]],
    result_columns[3]: [LGBM_result[3]]
}

df_results = pd.DataFrame(results_dict)
df_results.to_csv('model_results.csv', index=False)
print("結果已保存到 model_results.csv")