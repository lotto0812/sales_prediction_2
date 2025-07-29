import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
from joblib import dump
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

# データの準備（実際の使用時はすでにロードされたデータフレームを使用）
print("=== サンプルデータの作成 ===")
np.random.seed(42)
n_samples = 10000

# サンプルデータの作成
data = {
    'KEY': [f'restaurant_{i}' for i in range(n_samples)],
    'AVG_MONTHLY_POPULATION': np.random.randint(1000, 50000, n_samples),
    'RATING_CNT': np.random.randint(0, 5000, n_samples),
    'RATING_SCORE': np.random.uniform(3.0, 5.0, n_samples),
    'DINNER_INFO': np.random.choice([0, 1], n_samples),
    'LUNCH_INFO': np.random.choice([0, 1], n_samples),
    'HOME_PAGE_URL': np.random.choice([0, 1], n_samples),
    'PHONE_NUM': np.random.choice([0, 1], n_samples),
    'NUM_SEATS': np.random.randint(10, 200, n_samples),
    'MAX_NUM_PEOPLE_FOR_RESERVATION': np.random.randint(1, 50, n_samples),
    'RESERVATION_POSSIBILITY_INFO': np.random.choice([0, 1], n_samples),
    'CITY': np.random.choice([0, 1, 2, 3, 4], n_samples),
    'CUISINE_CAT_1': np.random.choice([0, 1, 2, 3, 4], n_samples)
}

# 目標変数の生成
target = (
    data['AVG_MONTHLY_POPULATION'] * 0.01 +
    data['RATING_CNT'] * 0.5 +
    data['RATING_SCORE'] * 10000 +
    data['NUM_SEATS'] * 100 +
    np.random.normal(0, 5000, n_samples)
)
data['average_target_amount_in_length_of_relationship'] = np.maximum(target, 0)

# データフレームの作成
df = pd.DataFrame(data)
print(f"データフレーム作成完了: {df.shape}")
print(f"目標変数の統計:\n{df['average_target_amount_in_length_of_relationship'].describe()}")

# 特徴量の定義
feature_columns = [
    'AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 'DINNER_INFO',
    'LUNCH_INFO', 'HOME_PAGE_URL', 'PHONE_NUM', 'NUM_SEATS', 
    'MAX_NUM_PEOPLE_FOR_RESERVATION', 'RESERVATION_POSSIBILITY_INFO',
    'CITY', 'CUISINE_CAT_1'
]
target_column = 'average_target_amount_in_length_of_relationship'

# 標準化する特徴量
standardize_columns = [
    'AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 
    'NUM_SEATS', 'MAX_NUM_PEOPLE_FOR_RESERVATION'
]

print(f"\n=== 特徴量の標準化 ===")
print(f"標準化対象: {standardize_columns}")

# 標準化処理
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[standardize_columns] = scaler.fit_transform(df[standardize_columns])

print("標準化完了")
print(f"標準化後の統計:\n{df_scaled[standardize_columns].describe()}")

# データの分割
print(f"\n=== データの分割 ===")
X = df_scaled[feature_columns]
y = df_scaled[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練データ: {X_train.shape}")
print(f"テストデータ: {X_test.shape}")

# ハイパーパラメータチューニング
print(f"\n=== ハイパーパラメータチューニング ===")
print("Optunaを使用してハイパーパラメータを最適化中...")

def objective(trial):
    """Optunaの目的関数"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'verbose': -1,
        'random_state': 42
    }
    
    # 訓練・検証データの分割
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    
    return rmse

def objective_ridge(trial):
    """Optunaの目的関数 for Ridge"""
    alpha = trial.suggest_float('alpha', 1e-8, 10.0, log=True)
    model = Ridge(alpha=alpha, random_state=42)
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model.fit(X_train_opt, y_train_opt)
    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

def objective_xgb(trial):
    """Optunaの目的関数 for XGBoost"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42
    }
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model = XGBRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              verbose=False)
    
    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

def objective_catboost(trial):
    """Optunaの目的関数 for CatBoost"""
    params = {
        'objective': 'RMSE',
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'random_state': 42
    }
    
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model = CatBoostRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              early_stopping_rounds=50,
              verbose=False)
    
    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

# Optunaでハイパーパラメータ最適化
print("\n--- LightGBM ---")
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective, n_trials=30)

print("\n--- Ridge ---")
study_ridge = optuna.create_study(direction='minimize')
study_ridge.optimize(objective_ridge, n_trials=30)

print("\n--- XGBoost ---")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30)

print("\n--- CatBoost ---")
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=30)


print(f"最適化完了!")
print(f"LGBM 最適なRMSE: {study_lgbm.best_value:.4f}")
print(f"LGBM 最適なパラメータ: {study_lgbm.best_params}")
print(f"Ridge 最適なRMSE: {study_ridge.best_value:.4f}")
print(f"Ridge 最適なパラメータ: {study_ridge.best_params}")
print(f"XGBoost 最適なRMSE: {study_xgb.best_value:.4f}")
print(f"XGBoost 最適なパラメータ: {study_xgb.best_params}")
print(f"CatBoost 最適なRMSE: {study_catboost.best_value:.4f}")
print(f"CatBoost 最適なパラメータ: {study_catboost.best_params}")

# 最適なパラメータでモデルを訓練
print(f"\n=== 最適なパラメータでモデル訓練 ===")
final_model = lgb.LGBMRegressor(**study_lgbm.best_params)
final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(50, verbose=False)])

# 予測
y_pred = final_model.predict(X_test)

# 評価指標の計算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== 最終モデルの評価結果 ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 他のモデルの訓練と評価
print(f"\n=== 他のモデルの訓練と評価 ===")

models = {
    'Ridge': Ridge(**study_ridge.best_params, random_state=42),
    'XGBoost': XGBRegressor(**study_xgb.best_params, random_state=42),
    'CatBoost': CatBoostRegressor(**study_catboost.best_params, random_state=42, verbose=0)
}

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    
    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred_model))
    mae_model = mean_absolute_error(y_test, y_pred_model)
    r2_model = r2_score(y_test, y_pred_model)
    
    print(f"RMSE: {rmse_model:.4f}")
    print(f"MAE: {mae_model:.4f}")
    print(f"R² Score: {r2_model:.4f}")

# アンサンブルモデル（スタッキング）
print(f"\n=== アンサンブルモデル（スタッキング）の訓練と評価 ===")
estimators = [
    ('lgbm', final_model),
    ('xgb', models['XGBoost']),
    ('cat', models['CatBoost'])
]

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=models['Ridge']
)

stacking_regressor.fit(X_train, y_train)
y_pred_stack = stacking_regressor.predict(X_test)

rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
mae_stack = mean_absolute_error(y_test, y_pred_stack)
r2_stack = r2_score(y_test, y_pred_stack)

print(f"RMSE: {rmse_stack:.4f}")
print(f"MAE: {mae_stack:.4f}")
print(f"R² Score: {r2_stack:.4f}")

# 特徴量重要度の可視化
print(f"\n=== 特徴量重要度の可視化 ===")

# LightGBM
lgbm_importance = final_model.feature_importances_
lgbm_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': lgbm_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=lgbm_importance_df, x='importance', y='feature')
plt.title('LightGBM 特徴量重要度')
plt.tight_layout()
plt.savefig('feature_importance_lgbm.png', dpi=300, bbox_inches='tight')
plt.show()

# XGBoost
xgb_model = models['XGBoost']
xgb_importance = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=xgb_importance_df, x='importance', y='feature')
plt.title('XGBoost 特徴量重要度')
plt.tight_layout()
plt.savefig('feature_importance_xgb.png', dpi=300, bbox_inches='tight')
plt.show()

# CatBoost
cat_model = models['CatBoost']
cat_importance = cat_model.get_feature_importance()
cat_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': cat_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=cat_importance_df, x='importance', y='feature')
plt.title('CatBoost 特徴量重要度')
plt.tight_layout()
plt.savefig('feature_importance_cat.png', dpi=300, bbox_inches='tight')
plt.show()

# 予測結果の可視化 (スタッキングモデル)
print(f"\n=== 予測結果の可視化 (スタッキングモデル) ===")
plt.figure(figsize=(12, 5))

# 予測値 vs 実際値
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_stack, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('実際値')
plt.ylabel('予測値')
plt.title('予測値 vs 実際値 (スタッキング)')

# 残差プロット
plt.subplot(1, 2, 2)
residuals_stack = y_test - y_pred_stack
plt.scatter(y_pred_stack, residuals_stack, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('予測値')
plt.ylabel('残差')
plt.title('残差プロット (スタッキング)')

plt.tight_layout()
plt.savefig('prediction_results_stacking.png', dpi=300, bbox_inches='tight')
plt.show()

# モデルの保存
print(f"\n=== モデルの保存 ===")
dump(final_model, 'lightgbm_sales_model.joblib')
print("LGBMモデルを保存しました: lightgbm_sales_model.joblib")
dump(stacking_regressor, 'stacking_sales_model.joblib')
print("スタッキングモデルを保存しました: stacking_sales_model.joblib")

# 最適化結果の可視化
print(f"\n=== 最適化結果の可視化 ===")
plt.figure(figsize=(12, 8))

plt.plot([trial.value for trial in study_lgbm.trials], label='LightGBM')
plt.plot([trial.value for trial in study_ridge.trials], label='Ridge')
plt.plot([trial.value for trial in study_xgb.trials], label='XGBoost')
plt.plot([trial.value for trial in study_catboost.trials], label='CatBoost')

plt.xlabel('試行回数')
plt.ylabel('RMSE')
plt.title('ハイパーパラメータ最適化の進行')
plt.legend()
plt.grid(True)
plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 完了 ===")
print("LightGBM, XGBoost, CatBoost, Ridge回帰モデルの作成とアンサンブルが完了しました。")
print("生成されたファイル:")
print("- lightgbm_sales_model.joblib: 訓練済みLGBMモデル")
print("- stacking_sales_model.joblib: 訓練済みスタッキングモデル")
print("- feature_importance_lgbm.png: LGBM特徴量重要度")
print("- feature_importance_xgb.png: XGBoost特徴量重要度")
print("- feature_importance_cat.png: CatBoost特徴量重要度")
print("- prediction_results_stacking.png: スタッキングモデルの予測結果")
print("- optimization_progress.png: 最適化の進行") 