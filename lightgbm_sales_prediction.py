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
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report, confusion_matrix


warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.linear_model import QuantileRegressor
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

# 目的変数の値を調整してhighクラスのサンプルを生成
high_class_indices = np.random.choice(range(n_samples), 10, replace=False)
data['average_target_amount_in_length_of_relationship'][high_class_indices] += 200000


# データフレームの作成
aggregated_df = pd.DataFrame(data)
print(f"データフレーム作成完了: {aggregated_df.shape}")
print(f"目標変数の統計:\n{aggregated_df['average_target_amount_in_length_of_relationship'].describe()}")

# ============================================================
# Ensemble Learning Implementation
# ============================================================
def main():
    # グローバル変数としてデータフレームを読み込む
    global aggregated_df

    feature_columns = [
        'AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 'DINNER_INFO',
        'LUNCH_INFO', 'HOME_PAGE_URL', 'PHONE_NUM', 'NUM_SEATS',
        'MAX_NUM_PEOPLE_FOR_RESERVATION', 'RESERVATION_POSSIBILITY_INFO',
        'CITY', 'CUISINE_CAT_1'
    ]
    target_column = 'average_target_amount_in_length_of_relationship'

    # ============================================================
    # 1. 前段の分類（ゲート）
    # ============================================================
    print("\n" + "="*60)
    print("1. Gating Classifier")
    print("="*60)

    # 1-1. ラベル化
    print("\n--- 1-1. Labeling ---")
    bins = [0, 50000, 150000, np.inf] # サンプルデータのスケールに合わせる
    labels = ['low', 'middle', 'high']
    aggregated_df['target_class'] = pd.cut(aggregated_df[target_column], bins=bins, labels=labels, right=False)
    
    # ラベルのエンコーディング
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    aggregated_df['target_class_encoded'] = le.fit_transform(aggregated_df['target_class'])
    
    print("Class distribution:\n", aggregated_df['target_class'].value_counts())

    # 1-2. 学習 (OOF)
    print("\n--- 1-2. Gating Classifier Training (OOF) ---")
    
    # 特徴量とターゲットの準備
    X = aggregated_df[feature_columns]
    y_class = aggregated_df['target_class_encoded']
    
    oof_gate_proba = np.zeros((len(aggregated_df), len(labels)))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
        print(f"  Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        oof_gate_proba[val_idx] = clf.predict_proba(X_val)
        
    # OOF予測をデータフレームに追加
    for i, label in enumerate(labels):
        aggregated_df[f'gate_proba_{label}'] = oof_gate_proba[:, i]
        
    print("OOF gate probabilities calculated and added to dataframe.")

    # 1-3. 評価（分類）
    print("\n--- 1-3. Gating Classifier Evaluation ---")
    oof_gate_pred = np.argmax(oof_gate_proba, axis=1)
    
    print("Classification Report (OOF):")
    print(classification_report(y_class, oof_gate_pred, target_names=labels))
    
    print("Confusion Matrix (OOF):")
    cm = confusion_matrix(y_class, oof_gate_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Gating Classifier Confusion Matrix (OOF)')
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 2. 各“専門家（回帰器）”の学習
    # ============================================================
    print("\n" + "="*60)
    print("2. Expert Regressors Training (OOF)")
    print("="*60)

    y_regr = aggregated_df[target_column]
    oof_preds = {}

    base_models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'Ridge': Ridge(random_state=42),
        'Tweedie': TweedieRegressor(power=1.5, link='log') # powerは仮置き
    }

    for name, model in base_models.items():
        print(f"\n--- Training {name} ---")
        oof_pred = np.zeros(len(aggregated_df))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
            print(f"  Fold {fold+1}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_regr.iloc[train_idx], y_regr.iloc[val_idx]
            
            # Tweedieは0以下の値を扱えないため調整
            if name == 'Tweedie':
                y_train = np.maximum(y_train, 1e-8)
                
            model.fit(X_train, y_train)
            oof_pred[val_idx] = model.predict(X_val)
            
        oof_preds[name] = oof_pred
        aggregated_df[f'oof_{name}'] = oof_pred
        
        rmse = np.sqrt(mean_squared_error(y_regr, oof_pred))
        print(f"  {name} OOF RMSE: {rmse:.4f}")

    # ============================================================
    # 3. Mixture（確率重みづけ）による第一次点予測
    # ============================================================
    print("\n" + "="*60)
    print("3. Mixture of Experts (Method A: Representative Values)")
    print("="*60)

    # 各ビンの代表値（中央値）を計算
    representative_values = aggregated_df.groupby('target_class', observed=True)[target_column].median()
    print("Representative Values (Median) for each class:\n", representative_values)

    # OOF確率と代表値でMixture予測を作成
    oof_mixture = (
        aggregated_df['gate_proba_low'] * representative_values['low'] +
        aggregated_df['gate_proba_middle'] * representative_values['middle'] +
        aggregated_df['gate_proba_high'] * representative_values['high']
    )
    aggregated_df['oof_mixture'] = oof_mixture
    
    rmse_mixture = np.sqrt(mean_squared_error(y_regr, oof_mixture))
    print(f"\nMixture Model OOF RMSE: {rmse_mixture:.4f}")

    # ============================================================
    # 4. Blending（重み付き平均）
    # ============================================================
    print("\n" + "="*60)
    print("4. Blending (Weighted Average with Pinball Loss Optimization)")
    print("="*60)

    # Pinball Loss関数
    def pinball_loss(y_true, y_pred, alpha=0.85):
        error = y_true - y_pred
        return np.mean(np.maximum(alpha * error, (alpha - 1) * error))

    # 最適化の目的関数
    def objective_for_blending(weights, oof_df, y_true):
        # 重みの合計が1になるように正規化（制約の一部）
        normalized_weights = weights / np.sum(weights)
        
        # OOF予測の加重平均を計算
        weighted_preds = np.sum(oof_df * normalized_weights, axis=1)
        
        return pinball_loss(y_true, weighted_preds)

    # ブレンド対象のOOF予測を抽出
    oof_columns = [col for col in aggregated_df.columns if col.startswith('oof_')]
    oof_df = aggregated_df[oof_columns]
    
    print("Blending models:", oof_columns)

    # 初期重み（均等）
    initial_weights = np.ones(len(oof_columns)) / len(oof_columns)

    # 制約条件: w_i >= 0, sum(w_i) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(oof_columns))]

    # 最適化の実行
    result = minimize(
        objective_for_blending,
        initial_weights,
        args=(oof_df, y_regr),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    print("\nOptimal Weights for Blending:")
    for name, weight in zip(oof_columns, optimal_weights):
        print(f"  {name}: {weight:.4f}")

    # 最適重みでブレンド予測を作成
    aggregated_df['oof_blend'] = np.sum(oof_df * optimal_weights, axis=1)
    
    rmse_blend = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_blend']))
    pinball_blend = pinball_loss(y_regr, aggregated_df['oof_blend'])
    
    print(f"\nBlend Model OOF RMSE: {rmse_blend:.4f}")
    print(f"Blend Model OOF Pinball Loss (alpha=0.85): {pinball_blend:.4f}")

    # ============================================================
    # 5. Stacking（メタ学習）
    # ============================================================
    print("\n" + "="*60)
    print("5. Stacking (Meta-Learning)")
    print("="*60)
    
    # メタ学習器の特徴量は、全ベースモデルとMixtureモデルのOOF予測
    meta_features = oof_columns
    X_meta = aggregated_df[meta_features]
    
    print("Meta-features for stacking:", meta_features)
    
    # メタ学習器（QuantileRegressor）をOOFで学習
    oof_stack = np.zeros(len(aggregated_df))
    meta_model = QuantileRegressor(quantile=0.85, alpha=0) # alpha=0は正則化なし
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_class)):
        print(f"  Fold {fold+1}")
        X_meta_train, X_meta_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
        y_train, y_val = y_regr.iloc[train_idx], y_regr.iloc[val_idx]
        
        meta_model.fit(X_meta_train, y_train)
        oof_stack[val_idx] = meta_model.predict(X_meta_val)
        
    aggregated_df['oof_stack'] = oof_stack
    
    rmse_stack = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_stack']))
    pinball_stack = pinball_loss(y_regr, aggregated_df['oof_stack'])
    
    print(f"\nStacking Model OOF RMSE: {rmse_stack:.4f}")
    print(f"Stacking Model OOF Pinball Loss (alpha=0.85): {pinball_stack:.4f}")

    # ============================================================
    # 6. 最終統合 & 微修正
    # ============================================================
    print("\n" + "="*60)
    print("6. Final Integration (Double Blend)")
    print("="*60)

    # 最終ブレンド対象のOOF予測を抽出
    final_blend_df = aggregated_df[['oof_blend', 'oof_stack']]
    
    # 初期重み
    initial_weights_final = np.array([0.5, 0.5])
    
    # 最適化の実行
    result_final = minimize(
        objective_for_blending,
        initial_weights_final,
        args=(final_blend_df, y_regr),
        method='SLSQP',
        bounds=[(0, 1), (0, 1)],
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    )

    optimal_weights_final = result_final.x
    print("\nOptimal Weights for Final Blend:")
    print(f"  oof_blend: {optimal_weights_final[0]:.4f}")
    print(f"  oof_stack: {optimal_weights_final[1]:.4f}")

    # 最終予測を作成
    aggregated_df['oof_final'] = (
        aggregated_df['oof_blend'] * optimal_weights_final[0] +
        aggregated_df['oof_stack'] * optimal_weights_final[1]
    )
    
    rmse_final = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_final']))
    pinball_final = pinball_loss(y_regr, aggregated_df['oof_final'])
    
    print(f"\nFinal Model OOF RMSE: {rmse_final:.4f}")
    print(f"Final Model OOF Pinball Loss (alpha=0.85): {pinball_final:.4f}")

    # ============================================================
    # 7. 評価 / モニタリング
    # ============================================================
    print("\n" + "="*60)
    print("7. Evaluation / Monitoring")
    print("="*60)
    
    # 予測 vs 実績プロット
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_regr, y=aggregated_df['oof_final'], alpha=0.5)
    plt.plot([y_regr.min(), y_regr.max()], [y_regr.min(), y_regr.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Final Predictions (OOF)')
    plt.title('True vs. Final Prediction')
    plt.tight_layout()
    plt.show()
    
    # 残差プロット
    residuals = y_regr - aggregated_df['oof_final']
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals (True - Pred)')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 8. 出力/成果物
    # ============================================================
    print("\n" + "="*60)
    print("8. Saving outputs")
    print("="*60)
    
    output_columns = [
        'KEY', 
        target_column, 
        'target_class',
        'gate_proba_low', 'gate_proba_middle', 'gate_proba_high',
        'oof_LightGBM', 'oof_CatBoost', 'oof_Ridge', 'oof_Tweedie',
        'oof_mixture', 'oof_blend', 'oof_stack', 'oof_final'
    ]
    output_df = aggregated_df[output_columns]
    output_df.to_csv('ensemble_oof_predictions.csv', index=False)
    print("OOF predictions saved to 'ensemble_oof_predictions.csv'")

if __name__ == "__main__":
    main()
print("\n=== 全ての処理が完了しました ===")
