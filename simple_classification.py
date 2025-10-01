import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("データを読み込み中...")
df = pd.read_excel('aggregated_df.xlsx')

# 分析用データの作成（20 < target_amount_tableau < 1000の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 1000)
df_analysis = df[analysis_condition].copy()

print(f"分析対象データ形状: {df_analysis.shape}")

# target_amount_tableauを4つのクラスに分類
df_analysis['y_class'] = pd.cut(
    df_analysis['target_amount_tableau'],
    bins=[20, 75, 150, 300, 1000],
    labels=['20~75', '75~150', '150~300', '300~'],
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

# 特徴量の標準化（SVM用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル定義（シンプルなパラメータ）
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'SVM': SVC(random_state=42, probability=True),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# モデル訓練と評価
results = {}

print(f"\n=== モデル訓練開始 ===")

for model_name, model in models.items():
    print(f"\n【{model_name}】")
    
    # モデルに応じてデータを選択
    if model_name == 'SVM':
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    # モデル訓練
    print("訓練中...")
    model.fit(X_train_model, y_train)
    
    # 予測
    y_pred = model.predict(X_test_model)
    y_pred_proba = model.predict_proba(X_test_model)
    
    # 精度を計算
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"テスト精度: {accuracy:.4f}")

# 結果の比較
print(f"\n=== モデル比較 ===")
print("-" * 40)
print(f"{'モデル':<15} {'テスト精度':<10}")
print("-" * 40)
for model_name, result in results.items():
    print(f"{model_name:<15} {result['accuracy']:<10.4f}")

# 最良のモデルを特定
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
print(f"\n最良のモデル: {best_model_name} (精度: {results[best_model_name]['accuracy']:.4f})")

# 詳細な分類レポート
print(f"\n=== {best_model_name} の詳細分類レポート ===")
print(classification_report(y_test, results[best_model_name]['y_pred']))

print(f"\n分析完了！")
