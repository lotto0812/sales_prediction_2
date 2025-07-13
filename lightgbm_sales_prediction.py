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
warnings.filterwarnings('ignore')

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
    
    # LightGBMデータセットの作成
    train_data = lgb.Dataset(X_train_opt, label=y_train_opt)
    val_data = lgb.Dataset(X_val_opt, label=y_val_opt, reference=train_data)
    
    # モデルの訓練
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # 予測と評価
    y_pred = model.predict(X_val_opt, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    
    return rmse

# Optunaでハイパーパラメータ最適化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"最適化完了!")
print(f"最適なRMSE: {study.best_value:.4f}")
print(f"最適なパラメータ: {study.best_params}")

# 最適なパラメータでモデルを訓練
print(f"\n=== 最適なパラメータでモデル訓練 ===")
best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42
})

# LightGBMデータセットの作成
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 最終モデルの訓練
final_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

# 予測
y_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# 評価指標の計算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== 最終モデルの評価結果 ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 特徴量重要度の可視化
print(f"\n=== 特徴量重要度の可視化 ===")
importance = final_model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='importance', y='feature')
plt.title('特徴量重要度')
plt.xlabel('重要度')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("特徴量重要度（上位10個）:")
print(importance_df.head(10))

# 予測結果の可視化
print(f"\n=== 予測結果の可視化 ===")
plt.figure(figsize=(12, 5))

# 予測値 vs 実際値
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('実際値')
plt.ylabel('予測値')
plt.title('予測値 vs 実際値')

# 残差プロット
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('予測値')
plt.ylabel('残差')
plt.title('残差プロット')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

# モデルの保存
print(f"\n=== モデルの保存 ===")
final_model.save_model('lightgbm_sales_model.txt')
print("モデルを保存しました: lightgbm_sales_model.txt")

# 最適化結果の可視化
print(f"\n=== 最適化結果の可視化 ===")
plt.figure(figsize=(10, 6))
plt.plot(range(len(study.trials)), [trial.value for trial in study.trials])
plt.xlabel('試行回数')
plt.ylabel('RMSE')
plt.title('ハイパーパラメータ最適化の進行')
plt.grid(True)
plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 完了 ===")
print("LightGBM回帰モデルの作成とハイパーパラメータチューニングが完了しました。")
print("生成されたファイル:")
print("- lightgbm_sales_model.txt: 訓練済みモデル")
print("- feature_importance.png: 特徴量重要度")
print("- prediction_results.png: 予測結果")
print("- optimization_progress.png: 最適化の進行") 