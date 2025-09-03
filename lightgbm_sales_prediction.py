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
import shap
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan # HDBSCANのインストールが必要な場合があります: pip install hdbscan
import argparse
import sys
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift, Birch
from sklearn.cluster import estimate_bandwidth
from tqdm import tqdm


warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import TweedieRegressor

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Sales Prediction with Clustering')
parser.add_argument('--clustering_method', type=str, default='kmeans',
                    choices=['kmeans', 'dbscan', 'hdbscan', 'gaussian_mixture'],
                    help='Clustering method to use (kmeans, dbscan, hdbscan, gaussian_mixture)')
args = parser.parse_args()
print(f"使用するクラスタリング手法: {args.clustering_method}")

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

# 特徴量のクラスタリング
print("\n=== 特徴量のクラスタリング ===")
clustering_features = ['AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 'NUM_SEATS']
X_for_clustering = df[clustering_features]

# クラスタリング用のデータを標準化
scaler_clustering = StandardScaler()
X_scaled_clustering = scaler_clustering.fit_transform(X_for_clustering)

# === クラスタリングのハイパーパラメータチューニング ===
print("\n=== クラスタリングのハイパーパラメータチューニング ===")

def objective_kmeans(trial, data):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(data, labels)

def objective_dbscan(trial, data):
    eps = trial.suggest_float('eps', 0.1, 2.0)
    min_samples = trial.suggest_int('min_samples', 2, 20)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    # クラスタ数が1以下（すべてノイズなど）の場合はスコアを-1とする
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    # ノイズポイントを除外して評価
    if -1 in unique_labels:
        filtered_indices = labels != -1
        # ノイズ除去後にクラスタが1つ以下になる場合
        if len(np.unique(labels[filtered_indices])) < 2:
            return -1.0
        return silhouette_score(data[filtered_indices], labels[filtered_indices])
    return silhouette_score(data, labels)

def objective_hdbscan(trial, data):
    min_cluster_size = trial.suggest_int('min_cluster_size', 5, 200)
    min_samples = trial.suggest_int('min_samples', 1, 20)
    cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0.0, 1.0)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, gen_min_span_tree=False)
    labels = model.fit_predict(data)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0
    if -1 in unique_labels:
        filtered_indices = labels != -1
        if len(np.unique(labels[filtered_indices])) < 2:
            return -1.0
        return silhouette_score(data[filtered_indices], labels[filtered_indices])
    return silhouette_score(data, labels)

def objective_gaussian_mixture(trial, data):
    n_components = trial.suggest_int('n_components', 2, 10)
    covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    labels = model.fit_predict(data)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(data, labels)

def objective_agglomerative(trial, data):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(data)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(data, labels)

def objective_birch(trial, data):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    threshold = trial.suggest_float('threshold', 0.1, 1.0)
    branching_factor = trial.suggest_int('branching_factor', 20, 100)
    model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    labels = model.fit_predict(data)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(data, labels)

def objective_meanshift(trial, data):
    bandwidth_estimate = estimate_bandwidth(data, quantile=0.2, n_samples=500)
    bandwidth = trial.suggest_float('bandwidth', bandwidth_estimate * 0.5, bandwidth_estimate * 1.5)
    model = MeanShift(bandwidth=bandwidth)
    labels = model.fit_predict(data)
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(data, labels)


# チューニングの実行とアルゴリズムのインスタンス化
print("\n=== チューニングされたパラメータでクラスタリングモデルを準備 ===")
tuned_clustering_algorithms = {}
n_trials = 10 # チューニングの試行回数を10に減らす

def run_study(objective_func, data, n_trials):
    study = optuna.create_study(direction='maximize')
    for _ in tqdm(range(n_trials), desc=f"Tuning {objective_func.__name__}"):
        study.optimize(lambda trial: objective_func(trial, data), n_trials=1)
    return study

# KMeans
study_kmeans = run_study(objective_kmeans, X_scaled_clustering, n_trials)
best_params_kmeans = study_kmeans.best_params
tuned_clustering_algorithms['KMeans'] = KMeans(**best_params_kmeans, random_state=42, n_init=10)
print(f"  Best params: {best_params_kmeans}, Silhouette: {study_kmeans.best_value:.4f}")

# DBSCAN
study_dbscan = run_study(objective_dbscan, X_scaled_clustering, n_trials)
best_params_dbscan = study_dbscan.best_params
tuned_clustering_algorithms['DBSCAN'] = DBSCAN(**best_params_dbscan)
print(f"  Best params: {best_params_dbscan}, Silhouette: {study_dbscan.best_value:.4f}")

# HDBSCAN
study_hdbscan = run_study(objective_hdbscan, X_scaled_clustering, n_trials)
best_params_hdbscan = study_hdbscan.best_params
tuned_clustering_algorithms['HDBSCAN'] = hdbscan.HDBSCAN(**best_params_hdbscan, gen_min_span_tree=True)
print(f"  Best params: {best_params_hdbscan}, Silhouette: {study_hdbscan.best_value:.4f}")

# GaussianMixture
study_gm = run_study(objective_gaussian_mixture, X_scaled_clustering, n_trials)
best_params_gm = study_gm.best_params
tuned_clustering_algorithms['GaussianMixture'] = GaussianMixture(**best_params_gm, random_state=42)
print(f"  Best params: {best_params_gm}, Silhouette: {study_gm.best_value:.4f}")

# AgglomerativeClustering
study_agg = run_study(objective_agglomerative, X_scaled_clustering, n_trials)
best_params_agg = study_agg.best_params
tuned_clustering_algorithms['AgglomerativeClustering'] = AgglomerativeClustering(**best_params_agg)
print(f"  Best params: {best_params_agg}, Silhouette: {study_agg.best_value:.4f}")

# MeanShift
study_ms = run_study(objective_meanshift, X_scaled_clustering, n_trials)
best_params_ms = study_ms.best_params
tuned_clustering_algorithms['MeanShift'] = MeanShift(**best_params_ms)
print(f"  Best params: {best_params_ms}, Silhouette: {study_ms.best_value:.4f}")

# Birch
study_birch = run_study(objective_birch, X_scaled_clustering, n_trials)
best_params_birch = study_birch.best_params
tuned_clustering_algorithms['Birch'] = Birch(**best_params_birch)
print(f"  Best params: {best_params_birch}, Silhouette: {study_birch.best_value:.4f}")


evaluation_results = {}
print("\n=== 全てのクラスタリングアルゴリズムを実行し、特徴量として追加 ===")
for name, algorithm in tqdm(tuned_clustering_algorithms.items(), desc="クラスタリング実行と評価"):
    column_name = f'cluster_{name.lower()}'
    print(f"\n--- クラスタリング手法: {name} (カラム名: {column_name}) ---")
    
    # クラスタリングの実行とカラム追加
    labels = algorithm.fit_predict(X_scaled_clustering)
    df[column_name] = labels
    
    # ノイズポイント（-1）を除外して評価
    if -1 in np.unique(labels):
        filtered_indices = labels != -1
        X_eval = X_scaled_clustering[filtered_indices]
        labels_eval = labels[filtered_indices]
    else:
        X_eval = X_scaled_clustering
        labels_eval = labels

    # クラスタ数が2以上の場合のみ評価指標を計算
    if len(np.unique(labels_eval)) > 1:
        silhouette = silhouette_score(X_eval, labels_eval)
        calinski = calinski_harabasz_score(X_eval, labels_eval)
        davies = davies_bouldin_score(X_eval, labels_eval)
        evaluation_results[name] = {
            'Silhouette Score': silhouette,
            'Calinski-Harabasz Score': calinski,
            'Davies-Bouldin Score': davies
        }
        print(f"  シルエットスコア: {silhouette:.4f}")
        print(f"  Calinski-Harabaszスコア: {calinski:.4f}")
        print(f"  Davies-Bouldinスコア: {davies:.4f}")
    else:
        print("  クラスタ数が1のため評価指標は計算しません。")
        evaluation_results[name] = {
            'Silhouette Score': 'N/A',
            'Calinski-Harabasz Score': 'N/A',
            'Davies-Bouldin Score': 'N/A'
        }
    
    print(f"  各クラスタのサンプル数:\n{df[column_name].value_counts().sort_index()}")
    
    # クラスタリング結果の可視化
    sns.pairplot(df, vars=clustering_features, hue=column_name, palette='viridis', plot_kws={'alpha': 0.6})
    plt.suptitle(f'Clustering Results Pair Plot ({name})', y=1.02)
    plt.savefig(f'clustering_results_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()


# 評価結果のサマリー
print("\n=== 全クラスタリング手法の評価結果サマリー ===")
evaluation_df = pd.DataFrame(evaluation_results).T
print(evaluation_df)

# print("\nクラスタリング評価が完了しました。ここで処理を終了します。")
# sys.exit()

# === クラスター別の目的変数ヒストグラムの作成 ===
print("\n=== クラスター別の目的変数ヒストグラムの作成 ===")
target_column_hist = 'average_target_amount_in_length_of_relationship'
cluster_columns_for_hist = [col for col in df.columns if col.startswith('cluster_')]

for col_name in cluster_columns_for_hist:
    print(f"--- {col_name} のヒストグラムを作成中 ---")
    
    # ノイズポイント(-1)を除外したデータフレームを作成
    plot_df = df[df[col_name] != -1].copy()
    
    if plot_df.empty or plot_df[col_name].nunique() == 0:
        print(f"  {col_name} には有効なクラスターがありません。スキップします。")
        continue

    plt.figure(figsize=(12, 8))
    
    # 各クラスターのヒストグラムを重ねて描画
    clusters = sorted(plot_df[col_name].unique())
    # 凡例が多すぎると見づらくなるため、20個までに制限
    if len(clusters) > 20:
        print(f"  警告: {col_name} のクラスター数が多すぎるため ({len(clusters)}個)、凡例は表示しません。")
        legend = False
    else:
        legend = True

    palette = sns.color_palette("viridis", n_colors=len(clusters))
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = plot_df[plot_df[col_name] == cluster_id]
        sns.histplot(cluster_data[target_column_hist], kde=True, label=f'Cluster {cluster_id}' if legend else None, alpha=0.6, color=palette[i])
        
    plt.title(f'Distribution of Target Amount by Cluster ({col_name})')
    plt.xlabel('Target Amount')
    plt.ylabel('Frequency')
    if legend:
        plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'histogram_target_by_cluster_{col_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nヒストグラムの作成が完了しました。")


# 特徴量の定義
# One-hot encode cluster columns
cluster_cols = [c for c in df.columns if c.startswith('cluster_')]
df = pd.get_dummies(df, columns=cluster_cols, prefix=cluster_cols, dtype=float)

# Adjust feature columns for regression
feature_columns = [
    'AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 'DINNER_INFO',
    'LUNCH_INFO', 'HOME_PAGE_URL', 'PHONE_NUM', 'NUM_SEATS',
    'MAX_NUM_PEOPLE_FOR_RESERVATION', 'RESERVATION_POSSIBILITY_INFO',
    'CITY', 'CUISINE_CAT_1'
]
dummy_cols = [c for c in df.columns if c.startswith('cluster_')]
feature_columns.extend(dummy_cols)

target_column = 'average_target_amount_in_length_of_relationship'

def calculate_mape(y_true, y_pred):
    """MAPE (Mean Absolute Percentage Error) を計算する"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 0で割るのを防ぐため、y_trueが0のデータは除外
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

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

# ハイパーパラメータチューニング用に訓練データをさらに分割
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ハイパーパラメータチューニング
print(f"\n=== ハイパーパラメータチューニング ===")
print("Optunaを使用してハイパーパラメータを最適化中...")

def objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
    """Optunaの目的関数 for LightGBM"""
    params = {
        'objective': 'quantile',
        'metric': 'rmse',
        'alpha': 0.8,
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

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              eval_metric='quantile',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))

    return rmse

def objective_ridge(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
    """Optunaの目的関数 for Ridge"""
    alpha = trial.suggest_float('alpha', 1e-8, 10.0, log=True)
    model = Ridge(alpha=alpha, random_state=42)

    model.fit(X_train_opt, y_train_opt)
    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

def objective_xgb(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
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

    model = XGBRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              verbose=False)

    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

def objective_catboost(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
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

    model = CatBoostRegressor(**params)
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              early_stopping_rounds=50,
              verbose=False)

    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

def objective_tabnet(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
    """Optunaの目的関数 for TabNet"""
    params = {
        'n_d': trial.suggest_int('n_d', 8, 32),
        'n_a': trial.suggest_int('n_a', 8, 32),
        'n_steps': trial.suggest_int('n_steps', 3, 7),
        'gamma': trial.suggest_float('gamma', 1.0, 1.5),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float("lr", 1e-2, 0.05, log=True)),
        'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        'scheduler_params': {"mode": "min", "patience": 5, "min_lr": 1e-5, "factor": 0.5},
        'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'verbose': 0,
        'seed': 42
    }

    model = TabNetRegressor(**params)

    # データ型をNumpy配列に変換
    X_train_np = X_train_opt.values
    y_train_np = y_train_opt.values.reshape(-1, 1)
    X_val_np = X_val_opt.values
    y_val_np = y_val_opt.values.reshape(-1, 1)

    model.fit(
        X_train=X_train_np, y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    y_pred = model.predict(X_val_np)
    rmse = np.sqrt(mean_squared_error(y_val_np, y_pred))

    return rmse

def objective_tweedie(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt):
    """Optunaの目的関数 for Tweedie GLM"""
    params = {
        'power': trial.suggest_float('power', 1.0, 2.0), # 1: Poisson, 2: Gamma
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True), # L2 regularization
        'link': 'log'
    }

    model = TweedieRegressor(**params)

    # y_train_optが0以下の値を含んでいると学習できないため、正の値にクリップする
    # Tweedie分布は正の値のみを扱うため
    y_train_opt_clipped = np.maximum(y_train_opt, 1e-8)

    model.fit(X_train_opt, y_train_opt_clipped)
    y_pred = model.predict(X_val_opt)
    rmse = np.sqrt(mean_squared_error(y_val_opt, y_pred))
    return rmse

# Optunaでハイパーパラメータ最適化
print("\n--- LightGBM ---")
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=30)

print("\n--- Ridge ---")
study_ridge = optuna.create_study(direction='minimize')
study_ridge.optimize(lambda trial: objective_ridge(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=30)

print("\n--- XGBoost ---")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=30)

print("\n--- CatBoost ---")
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(lambda trial: objective_catboost(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=30)

# print("\n--- TabNet ---")
# study_tabnet = optuna.create_study(direction='minimize')
# study_tabnet.optimize(lambda trial: objective_tabnet(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=15)

print("\n--- Tweedie GLM ---")
study_tweedie = optuna.create_study(direction='minimize')
study_tweedie.optimize(lambda trial: objective_tweedie(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=30)


print(f"\n最適化完了!")
print(f"LGBM 最適なRMSE: {study_lgbm.best_value:.4f}, パラメータ: {study_lgbm.best_params}")
print(f"Ridge 最適なRMSE: {study_ridge.best_value:.4f}, パラメータ: {study_ridge.best_params}")
print(f"XGBoost 最適なRMSE: {study_xgb.best_value:.4f}, パラメータ: {study_xgb.best_params}")
print(f"CatBoost 最適なRMSE: {study_catboost.best_value:.4f}, パラメータ: {study_catboost.best_params}")
# print(f"TabNet 最適なRMSE: {study_tabnet.best_value:.4f}, パラメータ: {study_tabnet.best_params}")
print(f"Tweedie GLM 最適なRMSE: {study_tweedie.best_value:.4f}, パラメータ: {study_tweedie.best_params}")

# 各モデルの訓練と評価
print(f"\n=== 各モデルの訓練と評価 ===")
results = {}

# LightGBM (final_modelとして別途訓練)
print("\n--- LightGBM ---")
final_model = lgb.LGBMRegressor(**study_lgbm.best_params,
                                objective='quantile',
                                alpha=0.8,
                                random_state=42)
final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='quantile',
                callbacks=[lgb.early_stopping(50, verbose=False)])
y_pred_lgbm = final_model.predict(X_test)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
mape_lgbm = calculate_mape(y_test, y_pred_lgbm)
results['LightGBM'] = {'RMSE': rmse_lgbm, 'MAE': mae_lgbm, 'MAPE': mape_lgbm, 'R2': r2_lgbm}
print(f"RMSE: {rmse_lgbm:.4f}, MAE: {mae_lgbm:.4f}, MAPE: {mape_lgbm:.2f}%, R²: {r2_lgbm:.4f}")

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
    mape_model = calculate_mape(y_test, y_pred_model)
    results[name] = {'RMSE': rmse_model, 'MAE': mae_model, 'MAPE': mape_model, 'R2': r2_model}
    print(f"RMSE: {rmse_model:.4f}, MAE: {mae_model:.4f}, MAPE: {mape_model:.2f}%, R²: {r2_model:.4f}")

# Tweedie GLM
print("\n--- Tweedie GLM ---")
tweedie_model = TweedieRegressor(**study_tweedie.best_params, link='log')
y_train_clipped = np.maximum(y_train, 1e-8)
tweedie_model.fit(X_train, y_train_clipped)
models['TweedieGLM'] = tweedie_model
y_pred_tweedie = tweedie_model.predict(X_test)
rmse_tweedie = np.sqrt(mean_squared_error(y_test, y_pred_tweedie))
mae_tweedie = mean_absolute_error(y_test, y_pred_tweedie)
r2_tweedie = r2_score(y_test, y_pred_tweedie)
mape_tweedie = calculate_mape(y_test, y_pred_tweedie)
results['TweedieGLM'] = {'RMSE': rmse_tweedie, 'MAE': mae_tweedie, 'MAPE': mape_tweedie, 'R2': r2_tweedie}
print(f"RMSE: {rmse_tweedie:.4f}, MAE: {mae_tweedie:.4f}, MAPE: {mape_tweedie:.2f}%, R²: {r2_tweedie:.4f}")

# # TabNet
# print("\n--- TabNet ---")
# best_params_tabnet = study_tabnet.best_params
# lr = best_params_tabnet.pop('lr')
# best_params_tabnet['optimizer_params'] = {'lr': lr}
# best_params_tabnet['optimizer_fn'] = torch.optim.Adam
# best_params_tabnet['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
# best_params_tabnet['scheduler_params'] = {"mode": "min", "patience": 10, "min_lr": 1e-5, "factor": 0.5}
# best_params_tabnet['seed'] = 42
# best_params_tabnet['verbose'] = 0
#
# tabnet_model = TabNetRegressor(**best_params_tabnet)
#
# X_train_np = X_train.values
# y_train_np = y_train.values.reshape(-1, 1)
# X_test_np = X_test.values
# y_test_np = y_test.values.reshape(-1, 1)
#
# tabnet_model.fit(
#     X_train=X_train_np, y_train=y_train_np,
#     eval_set=[(X_test_np, y_test_np)],
#     eval_metric=['rmse'],
#     max_epochs=200,
#     patience=50,
#     batch_size=1024,
#     virtual_batch_size=128,
#     num_workers=0,
#     drop_last=False,
# )
# models['TabNet'] = tabnet_model
# y_pred_tabnet = tabnet_model.predict(X_test.values)
# rmse_tabnet = np.sqrt(mean_squared_error(y_test, y_pred_tabnet))
# mae_tabnet = mean_absolute_error(y_test, y_pred_tabnet)
# r2_tabnet = r2_score(y_test, y_pred_tabnet)
# mape_tabnet = calculate_mape(y_test, y_pred_tabnet)
# results['TabNet'] = {'RMSE': rmse_tabnet, 'MAE': mae_tabnet, 'MAPE': mape_tabnet, 'R2': r2_tabnet}
# print(f"RMSE: {rmse_tabnet:.4f}, MAE: {mae_tabnet:.4f}, MAPE: {mape_tabnet:.2f}%, R²: {r2_tabnet:.4f}")

# # アンサンブルモデル（スタッキング）
# print(f"\n=== アンサンブルモデル（スタッキング）の訓練と評価 ===")
# # スタッキングのために、最適化されたパラメータで新しいモデルインスタンスを作成
# estimators = [
#     ('lgbm', lgb.LGBMRegressor(**study_lgbm.best_params, random_state=42)),
#     ('xgb', XGBRegressor(**study_xgb.best_params, random_state=42)),
#     ('cat', CatBoostRegressor(**study_catboost.best_params, random_state=42, verbose=0)),
#     ('tabnet', TabNetRegressor(**best_params_tabnet))
# ]

# stacking_regressor = StackingRegressor(
#     estimators=estimators,
#     final_estimator=Ridge(**study_ridge.best_params, random_state=42)
# )

# stacking_regressor.fit(X_train.values, y_train.values)
# y_pred_stack = stacking_regressor.predict(X_test.values)

# rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
# mae_stack = mean_absolute_error(y_test, y_pred_stack)
# r2_stack = r2_score(y_test, y_pred_stack)
# mape_stack = calculate_mape(y_test, y_pred_stack)
# results['Stacking'] = {'RMSE': rmse_stack, 'MAE': mae_stack, 'MAPE': mape_stack, 'R2': r2_stack}

# print(f"Stacking RMSE: {rmse_stack:.4f}, MAE: {mae_stack:.4f}, MAPE: {mape_stack:.2f}%, R²: {r2_stack:.4f}")


# 特徴量重要度の可視化
print(f"\n=== 特徴量重要度の可視化 ===")
all_models_for_importance = {
    'LightGBM': final_model,
    'XGBoost': models['XGBoost'],
    'CatBoost': models['CatBoost'],
    # 'TabNet': models['TabNet']
}

for name, model in all_models_for_importance.items():
    if name == 'TabNet':
        importance = model.feature_importances_
    elif name == 'CatBoost':
        importance = model.get_feature_importance()
    else:
        importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'{name} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Tweedie GLMの係数可視化
print("\n--- Tweedie GLM Coefficients ---")
tweedie_coeffs = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': models['TweedieGLM'].coef_
}).sort_values('coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=tweedie_coeffs, x='coefficient', y='feature')
plt.title('Tweedie GLM Coefficients')
plt.tight_layout()
plt.savefig('feature_importance_tweedieglm.png', dpi=300, bbox_inches='tight')
plt.show()


# SHAPの計算と可視化 (XGBoostモデルに対して)
print(f"\n=== SHAP値の計算と可視化 (XGBoost) ===")
explainer = shap.TreeExplainer(models['XGBoost'])
shap_values = explainer.shap_values(X_test)

# SHAPサマリープロット
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# 結果のサマリーと可視化
print("\n=== 全モデルの評価結果サマリー ===")
results_df = pd.DataFrame(results).T
print(results_df)

results_df.plot(kind='bar', y='RMSE', figsize=(12, 7), title='Model RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_rmse_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# モデルの保存
print(f"\n=== モデルの保存 ===")
dump(final_model, 'lightgbm_sales_model.joblib')
# dump(models['TabNet'], 'tabnet_sales_model.joblib')
dump(models['XGBoost'], 'xgboost_sales_model.joblib')
dump(models['CatBoost'], 'catboost_sales_model.joblib')
# dump(models['TabNet'], 'tabnet_sales_model.joblib')
dump(models['TweedieGLM'], 'tweedie_sales_model.joblib')

print("モデルが保存されました。")
print("\n=== 全ての処理が完了しました ===")
