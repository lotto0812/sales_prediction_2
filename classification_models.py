import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
print("データを読み込み中...")
df = pd.read_excel('aggregated_df.xlsx')

# 分析用データの作成（20 < target_amount_tableau < 1000の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 1000)
df_analysis = df[analysis_condition].copy()

print(f"分析対象データ形状: {df_analysis.shape}")
print(f"除外率: {(1 - len(df_analysis) / len(df)) * 100:.1f}%")

# target_amount_tableauを4つのクラスに分類
def create_y_class(df):
    """target_amount_tableauを4つのクラスに分類"""
    df['y_class'] = pd.cut(
        df['target_amount_tableau'],
        bins=[20, 75, 150, 300, 1000],
        labels=['20~75', '75~150', '150~300', '300~'],
        right=False
    )
    return df

# クラスを作成
df_analysis = create_y_class(df_analysis)

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
    'CUISINE_CAT_1_convenience store',
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

# 欠損値の確認
print(f"\n=== 欠損値の確認 ===")
missing_values = df_analysis[features].isnull().sum()
if missing_values.sum() > 0:
    print("欠損値がある特徴量:")
    for feature, missing_count in missing_values[missing_values > 0].items():
        print(f"  {feature}: {missing_count}個 ({missing_count/len(df_analysis)*100:.1f}%)")
else:
    print("欠損値なし")

# 欠損値を中央値で補完
df_analysis[features] = df_analysis[features].fillna(df_analysis[features].median())

# 特徴量とターゲットを準備
X = df_analysis[features]
y = df_analysis['y_class']

print(f"\n=== データ準備完了 ===")
print(f"特徴量数: {len(features)}")
print(f"サンプル数: {len(X)}")
print(f"クラス数: {len(np.unique(y))}")

# 訓練・テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== データ分割 ===")
print(f"訓練データ: {len(X_train)}サンプル")
print(f"テストデータ: {len(X_test)}サンプル")

# 特徴量の標準化（SVM用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル定義
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'SVM': SVC(random_state=42, probability=True),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
}

# ハイパーパラメータチューニング用のパラメータ
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# モデル訓練と評価
results = {}
best_models = {}

print(f"\n=== モデル訓練開始 ===")

for model_name, model in models.items():
    print(f"\n【{model_name}】")
    
    # ハイパーパラメータチューニング
    if model_name == 'SVM':
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    # GridSearchCVで最適なパラメータを探索
    grid_search = GridSearchCV(
        model, 
        param_grids[model_name], 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    print("ハイパーパラメータチューニング中...")
    grid_search.fit(X_train_model, y_train)
    
    # 最適なモデルを取得
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    print(f"最適パラメータ: {grid_search.best_params_}")
    print(f"CVスコア: {grid_search.best_score_:.4f}")
    
    # テストデータで予測
    y_pred = best_model.predict(X_test_model)
    y_pred_proba = best_model.predict_proba(X_test_model)
    
    # 精度を計算
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }
    
    print(f"テスト精度: {accuracy:.4f}")

# 結果の比較
print(f"\n=== モデル比較 ===")
print("-" * 60)
print(f"{'モデル':<15} {'CVスコア':<10} {'テスト精度':<10}")
print("-" * 60)
for model_name, result in results.items():
    print(f"{model_name:<15} {result['cv_score']:<10.4f} {result['accuracy']:<10.4f}")

# 最良のモデルを特定
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\n最良のモデル: {best_model_name} (精度: {results[best_model_name]['accuracy']:.4f})")

# 詳細な分類レポート
print(f"\n=== {best_model_name} の詳細分類レポート ===")
print(classification_report(y_test, results[best_model_name]['y_pred']))

# 混同行列の可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_counts.index, 
                yticklabels=class_counts.index,
                ax=axes[i])
    axes[i].set_title(f'{model_name}\n精度: {result["accuracy"]:.4f}')
    axes[i].set_xlabel('予測クラス')
    axes[i].set_ylabel('実際のクラス')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("混同行列を 'confusion_matrices.png' に保存しました。")

# 特徴量重要度の可視化（Random ForestとLightGBM）
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Random Forestの特徴量重要度
if 'Random Forest' in best_models:
    rf_importance = best_models['Random Forest'].feature_importances_
    rf_features_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_importance
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(x='importance', y='feature', data=rf_features_importance, ax=axes[0])
    axes[0].set_title('Random Forest 特徴量重要度 (Top 15)')
    axes[0].set_xlabel('重要度')

# LightGBMの特徴量重要度
if 'LightGBM' in best_models:
    lgb_importance = best_models['LightGBM'].feature_importances_
    lgb_features_importance = pd.DataFrame({
        'feature': features,
        'importance': lgb_importance
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(x='importance', y='feature', data=lgb_features_importance, ax=axes[1])
    axes[1].set_title('LightGBM 特徴量重要度 (Top 15)')
    axes[1].set_xlabel('重要度')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("特徴量重要度を 'feature_importance.png' に保存しました。")

# 結果をCSVに保存
print(f"\n=== 結果をCSVファイルに保存 ===")

# 各モデルの予測結果を保存
for model_name, result in results.items():
    pred_df = pd.DataFrame({
        'actual': y_test,
        'predicted': result['y_pred']
    })
    
    # 予測確率も追加
    for i, class_name in enumerate(class_counts.index):
        pred_df[f'prob_{class_name}'] = result['y_pred_proba'][:, i]
    
    filename = f'predictions_{model_name.replace(" ", "_")}.csv'
    pred_df.to_csv(filename, encoding='utf-8-sig', index=False)
    print(f"{model_name}の予測結果を '{filename}' に保存しました。")

# モデル比較結果を保存
comparison_df = pd.DataFrame({
    'モデル': list(results.keys()),
    'CVスコア': [results[model]['cv_score'] for model in results.keys()],
    'テスト精度': [results[model]['accuracy'] for model in results.keys()]
}).sort_values('テスト精度', ascending=False)

comparison_df.to_csv('model_comparison.csv', encoding='utf-8-sig', index=False)
print(f"モデル比較結果を 'model_comparison.csv' に保存しました。")

# 特徴量重要度を保存
if 'Random Forest' in best_models:
    rf_importance_df = pd.DataFrame({
        '特徴量': features,
        '重要度': best_models['Random Forest'].feature_importances_
    }).sort_values('重要度', ascending=False)
    rf_importance_df.to_csv('rf_feature_importance.csv', encoding='utf-8-sig', index=False)
    print(f"Random Forest特徴量重要度を 'rf_feature_importance.csv' に保存しました。")

if 'LightGBM' in best_models:
    lgb_importance_df = pd.DataFrame({
        '特徴量': features,
        '重要度': best_models['LightGBM'].feature_importances_
    }).sort_values('重要度', ascending=False)
    lgb_importance_df.to_csv('lgb_feature_importance.csv', encoding='utf-8-sig', index=False)
    print(f"LightGBM特徴量重要度を 'lgb_feature_importance.csv' に保存しました。")

print(f"\n分析完了！")
print(f"最良のモデル: {best_model_name}")
print(f"最高精度: {results[best_model_name]['accuracy']:.4f}")
