import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

print("データを読み込み中...")
df = pd.read_csv('aggregated_df_with_predictions.csv')

# 分析用データの作成（20 < target_amount_tableau < 1000の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 1000)
df_analysis = df[analysis_condition].copy()

print(f"分析対象データ形状: {df_analysis.shape}")

# target_amount_tableauを3つのクラスに分類
df_analysis['y_class'] = pd.cut(
    df_analysis['target_amount_tableau'],
    bins=[20, 50, 120, 1000],
    labels=['20~50', '50~120', '120~'],
    right=False
)

# クラス分布を確認
print("\n=== クラス分布 ===")
class_counts = df_analysis['y_class'].value_counts().sort_index()
for class_name, count in class_counts.items():
    percentage = count / len(df_analysis) * 100
    print(f"{class_name}: {count:,}店舗 ({percentage:.1f}%)")

# 特徴量を定義
features = [
    'AVG_MONTHLY_POPULATION',
    'NEAREST_STATION_INFO_count',
    'DISTANCE_VALUE',
    'RATING_CNT',
    'RATING_SCORE',
    'IS_FAMILY_FRIENDLY',
    'IS_FRIEND_FRIENDLY',
    'IS_ALONE_FRIENDLY',
    'DINNER_INFO',
    'LUNCH_INFO',
    'DINNER_PRICE',
    'LUNCH_PRICE',
    'HOLIDAY',
    'HOME_PAGE_URL',
    'PHONE_NUM',
    'NUM_SEATS',
    'AGE_RESTAURANT',
    'CITY_count',
    'CUISINE_CAT_1_bread and desert',
    'CUISINE_CAT_1_cafe and coffee shop',
    'CUISINE_CAT_1_cafeteria',
    'CUISINE_CAT_1_chinese cuisine',
    'CUISINE_CAT_1_family restaurant',
    'CUISINE_CAT_1_fastfood light meal',
    'CUISINE_CAT_1_foreign ethnic cuisine',
    'CUISINE_CAT_1_hotel and ryokan',
    'CUISINE_CAT_1_italian cuisine',
    'CUISINE_CAT_1_izakaya',
    'CUISINE_CAT_1_japanese cuisine',
    'CUISINE_CAT_1_noodles',
    'CUISINE_CAT_1_other',
    'CUISINE_CAT_1_restaurant',
    'rate_count',
    'seats_rate_count'
]

print(f"特徴量数: {len(features)}")

# 欠損値を中央値で補完
df_analysis[features] = df_analysis[features].fillna(df_analysis[features].median())

# 特徴量とターゲットを準備
X = df_analysis[features]
y = df_analysis['y_class']

print(f"\n=== データ準備完了 ===")
print(f"特徴量数: {len(features)}")
print(f"サンプル数: {len(X)}")

# 訓練・テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== データ分割 ===")
print(f"訓練データ: {len(X_train)}サンプル")
print(f"テストデータ: {len(X_test)}サンプル")

# 特徴量の標準化は不要（Random ForestとLightGBMは標準化不要）

# モデル定義とハイパーパラメータチューニング
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [20, 30],
            'min_samples_split': [5, 10]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [10, 15]
        }
    }
}

# モデル訓練と評価
results = {}

print(f"\n=== ハイパーパラメータ最適化開始 ===")
total_models = len(models)

for i, (model_name, model_config) in enumerate(models.items(), 1):
    print(f"\n【{i}/{total_models}】{model_name} の最適化開始")
    
    # データを選択（両モデルとも標準化なし）
    X_train_model = X_train
    X_test_model = X_test
    
    # グリッドサーチ実行
    print("ハイパーパラメータ最適化中...")
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['param_grid'],
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_model, y_train)
    
    # 最適なパラメータを表示
    print(f"最適パラメータ: {grid_search.best_params_}")
    print(f"最適スコア: {grid_search.best_score_:.4f}")
    
    # 最適モデルで予測
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_model)
    y_pred_proba = best_model.predict_proba(X_test_model)
    
    # 精度を計算
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': best_model
    }
    
    print(f"テスト精度: {accuracy:.4f}")
    
    # 特徴量重要度を取得・表示
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        feature_names = features
        
        # 重要度のDataFrameを作成
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Top10の特徴量重要度を表示
        print(f"\n=== {model_name} 特徴量重要度 Top10 ===")
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # 特徴量重要度のグラフを作成
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('重要度')
        plt.title(f'{model_name} 特徴量重要度 Top10')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特徴量重要度グラフを保存: feature_importance_{model_name.replace(' ', '_')}.png")
    
    # 各モデルの詳細分類レポートを表示
    print(f"\n=== {model_name} の詳細分類レポート ===")
    print(classification_report(y_test, y_pred))
    
    # 分類レポートをCSVファイルに保存
    class_report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    report_df.to_csv(f'classification_report_{model_name.replace(" ", "_")}.csv')
    print(f"分類レポートを保存: classification_report_{model_name.replace(' ', '_')}.csv")
    
    print(f"【{i}/{total_models}】{model_name} 完了")

# 結果の比較
print(f"\n=== 最適化モデル比較 ===")
print("-" * 80)
print(f"{'モデル':<15} {'テスト精度':<10} {'最適スコア':<10} {'最適パラメータ'}")
print("-" * 80)
for model_name, result in results.items():
    params_str = str(result['best_params'])[:30] + "..." if len(str(result['best_params'])) > 30 else str(result['best_params'])
    print(f"{model_name:<15} {result['accuracy']:<10.4f} {result['best_score']:<10.4f} {params_str}")

# 最良のモデルを特定
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\n最良のモデル: {best_model_name} (精度: {results[best_model_name]['accuracy']:.4f})")
print(f"最適パラメータ: {results[best_model_name]['best_params']}")
print(f"最適スコア: {results[best_model_name]['best_score']:.4f}")

# 各クラスのF1スコアを表示
class_report = classification_report(y_test, results[best_model_name]['y_pred'], output_dict=True)
print(f"\n=== 各クラスのF1スコア ===")
for class_name in class_counts.index:
    if class_name in class_report:
        f1 = class_report[class_name]['f1-score']
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        print(f"{class_name}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

print(f"\n分析完了！")

# 最良のモデルですべてのデータを分類
print(f"\n=== 全データの分類実行 ===")
print("最良のモデルですべてのデータを分類中...")

# 全データの特徴量を準備
X_all = df[features].fillna(df[features].median())
best_model = results[best_model_name]['best_model']

# 全データを分類
y_pred_all = best_model.predict(X_all)

# クラス名を英語に変換
class_mapping = {
    '20~50': 'low',
    '50~120': 'middle', 
    '120~': 'high'
}

pred_class = [class_mapping[class_name] for class_name in y_pred_all]

# aggregated_dfの一番右にpred_class列を追加
df['pred_class'] = pred_class

print(f"分類完了！")
print(f"全データ数: {len(df)}")
print(f"pred_class分布:")
pred_counts = df['pred_class'].value_counts()
for class_name, count in pred_counts.items():
    percentage = count / len(df) * 100
    print(f"  {class_name}: {count:,}店舗 ({percentage:.1f}%)")

# 結果をCSVファイルに保存
df.to_csv('aggregated_df_with_predictions.csv', index=False)
print(f"\n結果を保存: aggregated_df_with_predictions.csv")
