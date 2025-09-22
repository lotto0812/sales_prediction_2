import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def pinball_loss(y_true, y_pred, alpha=0.85):
    error = y_true - y_pred
    return np.mean(np.maximum(alpha * error, (alpha - 1) * error))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_model_evaluation(y_true, y_pred, model_name):
    """
    Generates four evaluation plots for a model's predictions in a 2x2 grid.
    1. True vs. Predicted scatter plot.
    2. Residuals vs. Predicted scatter plot.
    3. Distribution comparison histogram.
    4. Actual vs. Absolute Percentage Error scatter plot.
    """
    print(f"\n--- Evaluation Plots for {model_name} ---")
    
    valid_indices = y_true > 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    if len(y_true_valid) == 0:
        print(f"Skipping plots for {model_name} as there are no valid (y_true > 0) data points.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Evaluation Plots for {model_name}', fontsize=16)

    # Plot 1: True vs. Predicted
    sns.scatterplot(x=y_true_valid, y=y_pred_valid, alpha=0.5, ax=axes[0, 0])
    min_val = min(y_true_valid.min(), y_pred_valid.min())
    max_val = max(y_true_valid.max(), y_pred_valid.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_title('True vs. Predicted Values')
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].grid(True)

    # Plot 2: Residuals vs. Predicted
    residuals = y_true_valid - y_pred_valid
    sns.scatterplot(x=y_pred_valid, y=residuals, alpha=0.5, ax=axes[0, 1])
    axes[0, 1].axhline(0, color='r', linestyle='--')
    axes[0, 1].set_title('Residuals vs. Predicted Values')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals (True - Predicted)')
    axes[0, 1].grid(True)

    # Plot 3: Distribution Comparison
    sns.histplot(y_true_valid, color="blue", label='True Values', kde=True, stat="density", element="step", fill=False, ax=axes[1, 0])
    sns.histplot(y_pred_valid, color="red", label='Predicted Values', kde=True, stat="density", element="step", fill=False, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of True vs. Predicted Values')
    axes[1, 0].set_xlabel('Target Amount')
    axes[1, 0].legend()

    # Plot 4: Actual vs. Absolute Percentage Error
    abs_percentage_error = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
    sns.scatterplot(x=y_true_valid, y=abs_percentage_error, alpha=0.5, ax=axes[1, 1])
    axes[1, 1].set_title('Actual Values vs. Absolute Percentage Error')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Absolute Percentage Error (%)')
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def objective_for_blending(weights, Z, y_true, alpha):
    y_pred = np.dot(Z, weights)
    return pinball_loss(y_true, y_pred, alpha)

def main():
    # グローバル変数としてデータフレームを読み込む
    global aggregated_df

    # 特徴量とターゲットの定義
    feature_columns = [
        'AVG_MONTHLY_POPULATION','DISTANCE_VALUE','rate_count',
        "seats_rate_count","DINNER_PRICE","LUNCH_PRICE",
        'AGE_RESTAURANT','NEAREST_STATION_INFO_count','RATING_CNT',
        'RATING_SCORE','DINNER_INFO',
        'LUNCH_INFO','HOME_PAGE_URL','PHONE_NUM','NUM_SEATS',
        'CITY_count', 'CUISINE_CAT', 'CUISINE_CAT_origin'
    ]
    target_column = 'average_target_amount_in_length_of_relationship'

    # データの準備: aggregated_df.xlsx から読み込み
    print("\n=== データの読み込み ===")
    try:
        aggregated_df = pd.read_excel('aggregated_df.xlsx')
        print(f"データフレーム読み込み完了: {aggregated_df.shape}")
        
        # 外れ値の除去
        print("\n=== 外れ値の除去 ===")
        initial_rows = len(aggregated_df)
        print(f"除去前の行数: {initial_rows}")

        # 1. `target_column` > 1000 のデータを除去
        len_before_upper_clip = len(aggregated_df)
        aggregated_df = aggregated_df[aggregated_df[target_column] <= 1000]
        print(f"'{target_column}' > 1000 を除去: {len_before_upper_clip - len(aggregated_df)} 行")

        # 2. `target_amount_tableau` < 20 のデータを除去
        len_before_lower_clip = len(aggregated_df)
        aggregated_df = aggregated_df[aggregated_df['target_amount_tableau'] >= 30]
        print(f"'target_amount_tableau' < 20 を除去: {len_before_lower_clip - len(aggregated_df)} 行")

        # フィルタリング後のDataFrameのコピーを作成
        aggregated_df = aggregated_df.copy()

        final_rows = len(aggregated_df)
        print(f"最終的な行数: {final_rows} (合計 {initial_rows - final_rows} 行除去)")
        print(f"目標変数の統計 (除去後):\n{aggregated_df[target_column].describe()}")

    except FileNotFoundError:
        print("エラー: aggregated_df.xlsx が見つかりません。")
        print("スクリプトと同じディレクトリにファイルを配置してください。")
        sys.exit(1)


    # ============================================================
    # 0. 前処理 (数値特徴量の標準化)
    # ============================================================
    print("\n" + "="*60)
    print("0. Preprocessing (Standard Scaling)")
    print("="*60)
    
    # カテゴリカル特徴量をラベルエンコーディング
    categorical_cols = ['CITY', 'CUISINE_CAT', 'CUISINE_CAT_origin']
    for col in categorical_cols:
        if col in aggregated_df.columns:
            le = LabelEncoder()
            aggregated_df[col] = le.fit_transform(aggregated_df[col].astype(str))

    standardize_columns = [
        'AVG_MONTHLY_POPULATION', 'DISTANCE_VALUE', 'rate_count', 'seats_rate_count',
        'DINNER_PRICE', 'LUNCH_PRICE', 'AGE_RESTAURANT', 'NEAREST_STATION_INFO_count',
        'RATING_CNT', 'RATING_SCORE', 'NUM_SEATS', 'CITY_count'
    ]
    
    scaler = StandardScaler()
    # 標準化対象のカラムが存在するか確認してから実行
    existing_standardize_cols = [col for col in standardize_columns if col in aggregated_df.columns]
    aggregated_df[existing_standardize_cols] = scaler.fit_transform(aggregated_df[existing_standardize_cols])
    
    print("Numerical features standardized.")
    print(aggregated_df[existing_standardize_cols].describe())

    # ============================================================
    # 1. Gating Classifier
    # ============================================================
    print("\n" + "="*60)
    print("1. Gating Classifier")
    print("="*60)

    # 1-1. ラベル化
    print("\n--- 1-1. Labeling ---")
    bins = [0, 100, 200, np.inf] # 実際のデータ分布の仮定に合わせて変更
    labels = ['low', 'middle', 'high']
    # 目的変数の下限を0にクリップしてNaNを防ぐ
    target_clipped = aggregated_df[target_column].clip(0)
    aggregated_df['target_class'] = pd.cut(target_clipped, bins=bins, labels=labels, right=False)
    
    # ラベルのない行（NaN）を処理
    # 'target_class'がNaNの行を特定し、除去するか、特定のカテゴリに割り当てる
    # ここでは、最も一般的なクラス（'low'）に割り当てる例
    if aggregated_df['target_class'].isnull().any():
        print(f"Warning: Found {aggregated_df['target_class'].isnull().sum()} rows with NaN target_class. Filling with 'low'.")
        aggregated_df['target_class'].fillna('low', inplace=True)

    # ラベルのエンコーディング
    le_class = LabelEncoder()
    aggregated_df['target_class_encoded'] = le_class.fit_transform(aggregated_df['target_class'])
    
    print("Class distribution:\n", aggregated_df['target_class'].value_counts())

    # 1-2. 学習 (OOF)
    print("\n--- 1-2. Gating Classifier Training (OOF) ---")

    X = aggregated_df[feature_columns]
    y_class = aggregated_df['target_class_encoded']
    
    # カテゴリカル変数をPandasのカテゴリ型に変換
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    oof_gate_proba = np.zeros((len(aggregated_df), len(labels)))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
        print(f"  Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
        
        clf = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train, categorical_feature='auto')
        
        oof_gate_proba[val_idx] = clf.predict_proba(X_val)
        
    aggregated_df[['p_low', 'p_middle', 'p_high']] = oof_gate_proba
    print("OOF gate probabilities calculated and added to dataframe.")

    # 1-3. 評価
    print("\n--- 1-3. Gating Classifier Evaluation ---")
    oof_gate_pred_labels = np.argmax(oof_gate_proba, axis=1)
    
    # LabelEncoderのクラス順序とレポートのラベル順序を一致させる
    report_labels = le_class.inverse_transform(np.arange(len(labels)))

    print("Classification Report (OOF):")
    print(classification_report(y_class, oof_gate_pred_labels, target_names=report_labels))
    print("Confusion Matrix (OOF):")
    cm = confusion_matrix(y_class, oof_gate_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=report_labels, yticklabels=report_labels)
    plt.title('Confusion Matrix (OOF)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ============================================================
    # 2. Expert Regressors Training (OOF)
    # ============================================================
    print("\n" + "="*60)
    print("2. Expert Regressors Training (OOF)")
    print("="*60)

    y_regr = aggregated_df[target_column]
    # 対数変換した目的変数を用意 (NaNを防ぐためにclip)
    y_regr_log = np.log1p(y_regr.clip(0))
    
    oof_preds = {}

    base_models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'Ridge': Ridge(random_state=42),
        'Tweedie': TweedieRegressor(power=1.5, link='log')
    }

    for name, model in base_models.items():
        print(f"\n--- Training {name} ---")
        use_log_transform = name in ['LightGBM', 'CatBoost', 'Ridge']
        
        oof_pred_on_fit_scale = np.zeros(len(aggregated_df))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
            print(f"  Fold {fold+1}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

            if use_log_transform:
                y_train_regr = y_regr_log.iloc[train_idx]
            else:
                y_train_regr = y_regr.iloc[train_idx]

            if name == 'Tweedie':
                y_train_regr = np.maximum(y_train_regr, 1e-8)
                
            model.fit(X_train, y_train_regr)
            oof_pred_on_fit_scale[val_idx] = model.predict(X_val)
            
        if use_log_transform:
            oof_pred = np.expm1(oof_pred_on_fit_scale)
        else:
            oof_pred = oof_pred_on_fit_scale

        oof_preds[name] = oof_pred
        aggregated_df[f'oof_{name}'] = oof_pred
        
        rmse = np.sqrt(mean_squared_error(y_regr, oof_pred))
        print(f"  {name} OOF RMSE: {rmse:.4f}")
        
    # ============================================================
    # 3. Mixture of Experts (Method A: Representative Values)
    # ============================================================
    print("\n" + "="*60)
    print("3. Mixture of Experts (Method A: Representative Values)")
    print("="*60)

    class_medians = aggregated_df.groupby('target_class')[target_column].median()
    print("Representative Values (Median) for each class:")
    print(class_medians)

    # ゲート確率と代表値を掛け合わせて予測を作成
    # le_class.classes_でLabelEncoderが学習したクラス名を取得し、その順序で中央値を取得
    ordered_medians = class_medians[le_class.classes_].values
    oof_mixture = np.sum(oof_gate_proba * ordered_medians, axis=1)

    aggregated_df['oof_mixture'] = oof_mixture
    mixture_rmse = np.sqrt(mean_squared_error(y_regr, oof_mixture))
    print(f"\nMixture Model OOF RMSE: {mixture_rmse:.4f}")

    # ============================================================
    # 4. Blending (Weighted Average with Pinball Loss Optimization)
    # ============================================================
    print("\n" + "="*60)
    print("4. Blending (Weighted Average with Pinball Loss Optimization)")
    print("="*60)

    oof_columns_for_blending = [f'oof_{name}' for name in base_models.keys()] + ['oof_mixture']
    print(f"Blending models: {oof_columns_for_blending}\n")
    Z_blend = aggregated_df[oof_columns_for_blending].values

    initial_weights = np.ones(Z_blend.shape[1]) / Z_blend.shape[1]
    bounds = [(0, 1)] * Z_blend.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    result = minimize(
        objective_for_blending,
        initial_weights,
        args=(Z_blend, y_regr.values, 0.85),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights_blend = result.x
    print("Optimal Weights for Blending:")
    for name, weight in zip(oof_columns_for_blending, optimal_weights_blend):
        print(f"  {name}: {weight:.4f}")
        
    aggregated_df['oof_blend'] = np.dot(Z_blend, optimal_weights_blend)
    blend_rmse = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_blend']))
    blend_pinball = pinball_loss(y_regr, aggregated_df['oof_blend'])
    print(f"\nBlend Model OOF RMSE: {blend_rmse:.4f}")
    print(f"Blend Model OOF Pinball Loss (alpha=0.85): {blend_pinball:.4f}")

    # ============================================================
    # 5. Stacking (Meta-Learning)
    # ============================================================
    print("\n" + "="*60)
    print("5. Stacking (Meta-Learning)")
    print("="*60)

    meta_features = oof_columns_for_blending
    print(f"Meta-features for stacking: {meta_features}\n")
    X_meta = aggregated_df[meta_features]
    y_meta = y_regr

    oof_stack = np.zeros(len(aggregated_df))
    
    meta_model = QuantileRegressor(quantile=0.85, alpha=0, solver='highs')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
        print(f"  Fold {fold+1}")
        X_meta_train, X_meta_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
        y_meta_train, y_meta_val = y_meta.iloc[train_idx], y_meta.iloc[val_idx]

        meta_model.fit(X_meta_train, y_meta_train)
        oof_stack[val_idx] = meta_model.predict(X_meta_val)
        
    aggregated_df['oof_stack'] = oof_stack
    stack_rmse = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_stack']))
    stack_pinball = pinball_loss(y_regr, aggregated_df['oof_stack'])
    print(f"\nStacking Model OOF RMSE: {stack_rmse:.4f}")
    print(f"Stacking Model OOF Pinball Loss (alpha=0.85): {stack_pinball:.4f}")
    
    # ============================================================
    # 6. Final Integration (Double Blend)
    # ============================================================
    print("\n" + "="*60)
    print("6. Final Integration (Double Blend)")
    print("="*60)

    Z_final_blend = aggregated_df[['oof_blend', 'oof_stack']].values
    
    initial_weights_final = np.ones(Z_final_blend.shape[1]) / Z_final_blend.shape[1]
    bounds_final = [(0, 1)] * Z_final_blend.shape[1]
    constraints_final = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    result_final = minimize(
        objective_for_blending,
        initial_weights_final,
        args=(Z_final_blend, y_regr.values, 0.85),
        method='SLSQP',
        bounds=bounds_final,
        constraints=constraints_final
    )

    optimal_weights_final = result_final.x
    print("\nOptimal Weights for Final Blend:")
    print(f"  oof_blend: {optimal_weights_final[0]:.4f}")
    print(f"  oof_stack: {optimal_weights_final[1]:.4f}")
    
    aggregated_df['oof_final'] = np.dot(Z_final_blend, optimal_weights_final)
    final_rmse = np.sqrt(mean_squared_error(y_regr, aggregated_df['oof_final']))
    final_pinball = pinball_loss(y_regr, aggregated_df['oof_final'])
    final_mape = mean_absolute_percentage_error(y_regr, aggregated_df['oof_final'])

    print(f"\nFinal Model OOF RMSE: {final_rmse:.4f}")
    print(f"Final Model OOF MAPE: {final_mape:.4f} %")
    print(f"Final Model OOF Pinball Loss (alpha=0.85): {final_pinball:.4f}")
    
    # ============================================================
    # 7. 評価 / モニタリング
    # ============================================================
    print("\n" + "="*60)
    print("7. Evaluation / Monitoring")
    print("="*60)

    # 各シングルモデルとブレンドモデルの評価プロット
    models_to_evaluate = {
        'LightGBM': aggregated_df['oof_LightGBM'],
        'CatBoost': aggregated_df['oof_CatBoost'],
        'Ridge': aggregated_df['oof_Ridge'],
        'Tweedie': aggregated_df['oof_Tweedie'],
        'Blended Model': aggregated_df['oof_blend']
    }

    for name, y_pred in models_to_evaluate.items():
        plot_model_evaluation(y_regr, y_pred, name)

    # 最終モデルの評価プロット
    plot_model_evaluation(y_regr, aggregated_df['oof_final'], "Final Model")
    
    # ============================================================
    # 8. 出力/成果物
    # ============================================================
    print("\n" + "="*60)
    print("8. Saving outputs")
    print("="*60)

    output_columns = [
        target_column, 'target_class',
        'p_low', 'p_middle', 'p_high'
    ] + [f'oof_{name}' for name in base_models.keys()] + [
        'oof_mixture', 'oof_blend', 'oof_stack', 'oof_final'
    ]

    aggregated_df[output_columns].to_csv('ensemble_oof_predictions.csv', index=False)
    print("OOF predictions saved to 'ensemble_oof_predictions.csv'")

    print("\n=== 全ての処理が完了しました ===")

if __name__ == "__main__":
    main()
