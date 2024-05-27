"""匯入模組"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, RocCurveDisplay
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

df = pd.read_excel("US Superstore data.xlsx")

"""移除以下特徵"""

sample = df.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Product ID', 'Customer Name', 'Country', 'Ship Date', 'Postal Code', 'Product ID', 'Product Name', 'Profit', 'State', 'Category'])

"""把日期換成季度"""

order_date = df['Order Date']
order_season = []
for i in order_date:
    if i.month in [1, 2, 3]:
        order_season.append('1')
    elif i.month in [4, 5, 6]:
        order_season.append('2')
    elif i.month in [7, 8, 9]:
        order_season.append('3')
    elif i.month in [10, 11, 12]:
        order_season.append('4')
sample['order_season'] = order_season
sample = sample.drop(columns=['Order Date'])

"""把Ship Mode換成序數特徵"""

ship_mode = df['Ship Mode']
ship_degree = []
for i in ship_mode:
    if i == 'Standard Class':
        ship_degree.append(0)
    elif i == 'Second Class':
        ship_degree.append(1)
    elif i == 'First Class':
        ship_degree.append(2)
    elif i == 'Same Day':
        ship_degree.append(3)
sample['Ship Mode'] = ship_degree

"""分割樣本 8:2  訓練:測試"""

train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)

"""把數量特徵正規化成Z分數，把類別變數轉換成 hot code格式"""

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = ['Ship Mode', 'Quantity', 'Discount', 'order_season']
cat_attribs = ['Segment', 'City', 'Region', 'Sub-Category']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])
sample_prepared = full_pipeline.fit_transform(sample)
train_set_prepared = full_pipeline.transform(train_set)
y_train = train_set['Sales']

"""全部參數(8個)之決策樹迴歸"""

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_set_prepared, y_train)
print("全部參數(8個)之決策樹迴歸")
#樣本內表現
train_prediction = tree_reg.predict(train_set_prepared)
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = tree_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#特徵重要性分析
print('特徵重要性分析')
importances = tree_reg.feature_importances_
feature_names = full_pipeline.get_feature_names_out()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df.head(542))
plt.barh(importance_df.Feature[:20], importance_df.Importance[:20])
plt.show()
#儲存模型
joblib.dump(tree_reg, 'model/tree_reg.pkl')

"""全部參數(8個)之隨機森林迴歸"""

forest_reg = RandomForestRegressor()
forest_reg.fit(train_set_prepared, y_train)
print("全部參數(8個)之隨機森林迴歸")
#樣本內表現
train_prediction = forest_reg.predict(train_set_prepared)
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = forest_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#儲存模型
joblib.dump(forest_reg, 'model/forest_reg.pkl')

"""全部參數(8個)之SVM迴歸"""

svm_reg = LinearSVR(epsilon=0.1)
svm_reg.fit(train_set_prepared, y_train)
print("全部參數(8個)之SVM迴歸")
#樣本內表現
train_prediction = svm_reg.predict(train_set_prepared)
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = svm_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#儲存模型
joblib.dump(svm_reg, 'model/svm_reg.pkl')

"""全部參數(8個)之XGBoost迴歸"""

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("全部參數(8個)之XGBoost迴歸")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#儲存模型
joblib.dump(xgb_reg, 'model/xgb_reg.pkl')

"""XGBoost超參數最佳化"""

param_grid = {
    'n_estimators': [20, 40, 60, 80],
    'learning_rate': [0.5, 1, 1.5, 2],
    'max_depth': [3, 4, 5],
    'gamma': [0.1, 0.3, 0.5]
}
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(train_set_prepared, y_train)
# 輸出最佳參數
print("Best parameters found: ", grid_search.best_params_)

"""全部參數(8個)之最佳化XGBoot"""

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.5, gamma=0.1)
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("全部參數(8個)之最佳化XGBoot")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#儲存模型
joblib.dump(xgb_reg, 'model/xgb_best_reg.pkl')

"""移除 region"""

sample = sample.drop(columns=['Region'])
train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)
y_train = train_set['Sales']

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = ['Ship Mode', 'Quantity', 'Discount', 'order_season']
cat_attribs = ['Segment', 'City', 'Sub-Category']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
])
sample_prepared = full_pipeline.fit_transform(sample)
train_set_prepared = full_pipeline.transform(train_set)

"""移除1個特徵之XGBoost"""

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除1個特徵之XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#儲存模型
joblib.dump(xgb_reg, 'model/xgb-1_reg.pkl')

"""最佳化"""

param_grid = {
    'n_estimators': [20, 40, 60, 80],
    'learning_rate': [0.5, 1, 1.5, 2],
    'max_depth': [3, 4, 5],
    'gamma': [0.1, 0.3, 0.5]
}
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(train_set_prepared, y_train)

# 輸出最佳參數
print("Best parameters found: ", grid_search.best_params_)

"""移除1個特徵之最佳XGBoost"""

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.5, gamma=0.1)
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除1個特徵之最佳XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-1_best_reg.pkl')

"""畫學習曲線"""

train_sizes, train_scores, validation_scores = learning_curve(xgb_reg, train_set_prepared, y_train, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], scoring = 'neg_root_mean_squared_error')
# 計算平均值和標準差
train_scores_mean = -np.mean(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.legend(loc="best")
plt.show()

"""移除 discount"""

sample = sample.drop(columns=['Discount'])
sample

train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)
y_train = train_set['Sales']

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = ['Ship Mode', 'Quantity', 'order_season']
cat_attribs = ['Segment', 'City', 'Sub-Category']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
])
sample_prepared = full_pipeline.fit_transform(sample)
train_set_prepared = full_pipeline.transform(train_set)

"""移除2個特徵之XGBoost"""

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除2個特徵之XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-2_reg.pkl')

"""參數最佳化"""

param_grid = {
    'n_estimators': [20, 40, 60, 80],
    'learning_rate': [0.5, 1, 1.5, 2],
    'max_depth': [3, 4, 5],
    'gamma': [0.1, 0.3, 0.5]
}
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(train_set_prepared, y_train)

# 輸出最佳參數
print("Best parameters found: ", grid_search.best_params_)

"""移除2個特徵之最佳XGBoost"""

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=1, gamma=0.1)
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除2個特徵之最佳XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-2_best_reg.pkl')

"""畫學習曲線"""

train_sizes, train_scores, validation_scores = learning_curve(xgb_reg, train_set_prepared, y_train, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], scoring = 'neg_root_mean_squared_error')
# 計算平均值和標準差
train_scores_mean = -np.mean(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.legend(loc="best")
plt.show()

"""移除 Segment"""

sample = sample.drop(columns=['Segment'])
train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)
y_train = train_set['Sales']

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = ['Ship Mode', 'Quantity', 'order_season']
cat_attribs = ['City', 'Sub-Category']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])
sample_prepared = full_pipeline.fit_transform(sample)
train_set_prepared = full_pipeline.transform(train_set)

"""移除3個特徵之XGBoost"""

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除3個特徵之XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-3_reg.pkl')

"""參數最佳化"""

param_grid = {
    'n_estimators': [20, 40, 60, 80],
    'learning_rate': [0.5, 1, 1.5, 2],
    'max_depth': [3, 4, 5],
    'gamma': [0.1, 0.3, 0.5]
}
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(train_set_prepared, y_train)

#輸出最佳參數
print("Best parameters found: ", grid_search.best_params_)

"""移除3個特徵之最佳XGBoost"""

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=1, gamma=0.1)
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除3個特徵之最佳XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-3_best_reg.pkl')

"""畫學習曲線"""

train_sizes, train_scores, validation_scores = learning_curve(xgb_reg, train_set_prepared, y_train, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], scoring = 'neg_root_mean_squared_error')
# 計算平均值和標準差
train_scores_mean = -np.mean(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.legend(loc="best")
plt.show()

"""移除 Ship Mode"""

sample = sample.drop(columns=['Ship Mode'])
train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)
y_train = train_set['Sales']

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = ['Quantity', 'order_season']
cat_attribs = ['City', 'Sub-Category']
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])
sample_prepared = full_pipeline.fit_transform(sample)
train_set_prepared = full_pipeline.transform(train_set)

"""移除4個特徵之XGBoost"""

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除4個特徵之XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-4_reg.pkl')

"""參數最佳化"""

param_grid = {
    'n_estimators': [20, 40, 60, 80],
    'learning_rate': [0.5, 1, 1.5, 2],
    'max_depth': [3, 4, 5],
    'gamma': [0.1, 0.3, 0.5]
}
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5)
grid_search.fit(train_set_prepared, y_train)

#輸出最佳超參數
print("Best parameters found: ", grid_search.best_params_)

"""移除4個特徵之最佳XGBoost"""

xgb_reg = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=1, gamma=0.1)
xgb_reg.fit(train_set_prepared, y_train)
train_prediction = xgb_reg.predict(train_set_prepared)
print("移除4個特徵之最佳XGBoost")
#樣本內表現
MAE = mean_absolute_error(y_train, train_prediction)
MSE = mean_squared_error(y_train, train_prediction)
RMSE = mean_squared_error(y_train, train_prediction, squared=False)
print('樣本內表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')
#樣本外表現
y_test = test_set['Sales'].copy()
x_test_prepared = full_pipeline.transform(test_set)
test_prediction = xgb_reg.predict(x_test_prepared)

MAE = mean_absolute_error(y_test, test_prediction)
MSE = mean_squared_error(y_test, test_prediction)
RMSE = mean_squared_error(y_test, test_prediction, squared=False)
print('樣本外表現')
print(f'MAE:{MAE} MSE:{MSE} RMSE:{RMSE}')

joblib.dump(xgb_reg, 'model/xgb-4_best_reg.pkl')

"""畫學習曲線"""

train_sizes, train_scores, validation_scores = learning_curve(xgb_reg, train_set_prepared, y_train, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], scoring = 'neg_root_mean_squared_error')
# 計算平均值和標準差
train_scores_mean = -np.mean(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# 繪製學習曲線
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.legend(loc="best")
plt.show()