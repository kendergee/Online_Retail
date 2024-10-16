import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
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
from sklearn.model_selection import learning_curve
import optuna
import xgboost as xgb


data = pd.read_csv(r'E:\vscode\Online_Retail\online_retail_II.csv')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Year'] = data['InvoiceDate'].dt.year
data['Quarter'] =data['InvoiceDate'].dt.quarter
data = data.drop(['Invoice','Description','InvoiceDate','Customer ID'],axis=1)
data = data.groupby(['Country','StockCode','Year','Quarter'])['Quantity'].sum().reset_index()
data['Rolling_Avg_Sales_3'] = (data.groupby(['Country', 'StockCode'])['Quantity'].transform(lambda x: x.rolling(window=3, min_periods=1).mean()))
# data['Sales_Growth_Rate'] = (data.groupby(['Country', 'StockCode'])['Quantity'].transform(lambda x: x.pct_change()))
# data['Sales_Growth_Rate'].fillna(0, inplace=True)

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
    param_grid = {
    'n_estimators': [300, 700,1000,1500],
    'max_depth': [None,100],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(
        estimator=RF, 
        param_grid=param_grid,
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=6
    )
    grid_search.fit(X_train, y_train)
    print("最佳參數:", grid_search.best_params_)
    print("最佳分數:", -grid_search.best_score_)
    best_model = grid_search.best_estimator_

    train_sizes, train_scores, val_scores = learning_curve(best_model, X, y, cv=5, scoring='neg_mean_squared_error',n_jobs=6)
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Error')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Error')
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('RF learing curve', dpi=300, bbox_inches='tight') 
    plt.clf()
    
    y_pred = best_model.predict(X_test)

    # 計算 MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test,y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('RF結束')

    RF_result = {'best param':best_model.get_params(),
                 'best cv score':-grid_search.best_score_,
                 'Test mse':mse,
                 'Test R2':r2}
    df_results = pd.DataFrame(RF_result)
    df_results.to_csv('model_results RF.csv', index=False)
    print("結果已保存到 model_results RF.csv")

    return RF_result

def objective(trial):
    param = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',  # 確保 GPU 設置
    'device': 'cuda',       # 指定 GPU 設備
    'max_depth': trial.suggest_int('max_depth', 3, 20),  # 增加到 20
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),  # 擴大範圍，0.001 - 0.5
    'subsample': trial.suggest_float('subsample', 0.3, 1.0, step=0.05),  # 增加範圍到 0.3 - 1.0
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0, step=0.05),  # 增加範圍到 0.3 - 1.0
    'reg_alpha': trial.suggest_categorical('reg_alpha', [0, 0.1, 0.5, 1, 2, 5, 10]),  # 增加到 10
    'reg_lambda': trial.suggest_categorical('reg_lambda', [0.1, 0.5, 1, 2, 3, 5, 10, 20]),  # 增加到 20
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # 增加最小葉子節點的權重
    'gamma': trial.suggest_float('gamma', 0, 10.0, step=0.5),  # 添加 gamma 參數
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0, step=0.5)  # 添加類別不平衡時的縮放權重
    }

    # 不在 DMatrix 中指定設備
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 使用 XGBoost 的交叉驗證功能
    cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        nfold=5,
        num_boost_round=trial.suggest_categorical('n_estimators', [500, 1000, 1500, 2000, 3000, 5000]),  # 增加到 5000
        metrics='rmse',
        as_pandas=True,
        seed=42
    )
    
    return cv_results['test-rmse-mean'].min()

def XGBoost(X_train, X_test, y_train, y_test):
    print('XGB開始')

    # 使用 Optuna 進行超參數調整
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)  # 可以調整試驗次數

    best_params = study.best_params
    print("最佳參數:", best_params)

    # 使用最佳參數創建模型
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',  # 設置 GPU 設備
        n_estimators=best_params.pop('n_estimators'),
        **best_params
    )
    best_model.fit(X_train, y_train)

    # 繪製學習曲線
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Error')
    plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation Error')
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('XGB_learning_curve.png', dpi=300, bbox_inches='tight') 
    plt.clf()

    # 使用 GPU 進行預測
    y_pred = best_model.predict(X_test)

    # 計算 MSE 和 R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test r2: {r2}")
    print('XGB結束')

    XGB_result = {
        'best param': best_params,
        'best cv score': study.best_value,
        'Test mse': mse,
        'Test R2': r2
    }
    df_results = pd.DataFrame([XGB_result])
    df_results.to_csv('model_results_XGB.csv', index=False)
    print("結果已保存到 model_results_XGB.csv")
    joblib.dump(best_model,'Stock Prediction model.pkl')

    return XGB_result



def GBR(X_train,X_test,y_train,y_test):
    print('GBR開始')
    GBR = GradientBoostingRegressor()
    cv_scores = cross_val_score(GBR,X_train,y_train,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
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
    cv_scores = cross_val_score(LGBM,X_train,y_train,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
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

# RF_result = RF(X_train,X_test,y_train,y_test)
XGB_result = XGBoost(X_train,X_test,y_train,y_test)
# GBR_result = GBR(X_train,X_test,y_train,y_test)
# LGBM_result = LGBM(X_train,X_test,y_train,y_test)

# results_dict = {
#     'Model': ['RF','XGB','GBR','LightGBM'],
#     result_columns[0]: [RF_result[0],XGB_result[0],GBR_result[0],LGBM_result[0]],
#     result_columns[1]: [RF_result[1],XGB_result[1],GBR_result[1],LGBM_result[1]],
#     result_columns[2]: [RF_result[2],XGB_result[2],GBR_result[2],LGBM_result[2]],
#     result_columns[3]: [RF_result[3],XGB_result[3],GBR_result[3],LGBM_result[3]],
#     result_columns[4]: [RF_result[4],XGB_result[4],GBR_result[4],LGBM_result[4]],
# }

# df_results = pd.DataFrame(results_dict)
# df_results.to_csv('model_results 2.csv', index=False)
# print("結果已保存到 model_results 2.csv")