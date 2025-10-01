import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
print("データを読み込み中...")
df = pd.read_excel('aggregated_df.xlsx')

print(f"元データ形状: {df.shape}")

# target_amount_tableauの範囲を作成
def create_amount_ranges(df):
    """target_amount_tableauの範囲を作成"""
    df['amount_range'] = pd.cut(
        df['target_amount_tableau'],
        bins=[-np.inf, 20, 50, 150, 300, 600, 1000, np.inf],
        labels=['<20', '20~50', '50~150', '150~300', '300~600', '600~1000', '1000~'],
        right=False
    )
    return df

# 範囲を作成
df = create_amount_ranges(df)

# 分析用データの作成（20 < target_amount_tableau < 600の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 600)
df_analysis = df[analysis_condition].copy()

print(f"分析対象データ形状: {df_analysis.shape}")

# CUISINE_CAT_originの分布を確認
print(f"\nCUISINE_CAT_originの分布:")
cuisine_counts = df_analysis['CUISINE_CAT_origin'].value_counts()
print(cuisine_counts)

# データ数が少ないカテゴリを除外（最低50店舗以上）
min_count = 50
valid_cuisines = cuisine_counts[cuisine_counts >= min_count].index
print(f"\n分析対象の料理カテゴリ（{min_count}店舗以上）:")
print(f"対象カテゴリ数: {len(valid_cuisines)}")
print(f"対象データ数: {df_analysis[df_analysis['CUISINE_CAT_origin'].isin(valid_cuisines)].shape[0]}")

# 統計情報を計算する関数
def calculate_percentile_stats(series):
    return {
        'データ数': len(series),
        '平均': series.mean(),
        '標準偏差': series.std(),
        'min': series.min(),
        '10%タイル': series.quantile(0.1),
        '25%タイル': series.quantile(0.25),
        '中央値': series.median(),
        '75%タイル': series.quantile(0.75),
        '90%タイル': series.quantile(0.9),
        'max': series.max()
    }

# 各料理カテゴリでクラスタリングを実行
results = {}
cluster_stats_all = {}

print(f"\n=== 各料理カテゴリでのクラスタリング分析 ===")

for cuisine in valid_cuisines:
    print(f"\n【{cuisine}】")
    
    # 該当カテゴリのデータを抽出
    cuisine_data = df_analysis[df_analysis['CUISINE_CAT_origin'] == cuisine].copy()
    print(f"データ数: {len(cuisine_data):,}店舗")
    
    if len(cuisine_data) < 10:  # データ数が少なすぎる場合はスキップ
        print("データ数が少なすぎるためスキップ")
        continue
    
    # クラスタリング用の特徴量を準備
    clustering_features = ['AVG_MONTHLY_POPULATION', 'NUM_SEATS']
    X = cuisine_data[clustering_features].copy()
    
    # 欠損値の確認
    if X.isnull().sum().sum() > 0:
        print("欠損値があるためスキップ")
        continue
    
    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # クラスタ数3でクラスタリング実行
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # シルエットスコアを計算
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"シルエットスコア: {silhouette_avg:.3f}")
    
    # クラスタラベルをデータフレームに追加
    cuisine_data['cluster'] = cluster_labels
    
    # 各クラスタの基本情報
    print(f"各クラスタのデータ数:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for i, count in cluster_counts.items():
        print(f"  クラスタ {i}: {count:,}店舗")
    
    # 各クラスタの統計情報を計算
    cluster_stats = {}
    for cluster_id in range(3):
        cluster_data = cuisine_data[cuisine_data['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            # target_amount_tableauの統計
            target_stats = calculate_percentile_stats(cluster_data['target_amount_tableau'])
            # 特徴量の統計
            feature_stats = {
                '人流平均': cluster_data['AVG_MONTHLY_POPULATION'].mean(),
                '席数平均': cluster_data['NUM_SEATS'].mean(),
                '人流中央値': cluster_data['AVG_MONTHLY_POPULATION'].median(),
                '席数中央値': cluster_data['NUM_SEATS'].median()
            }
            
            cluster_stats[cluster_id] = {**target_stats, **feature_stats}
            
            print(f"  クラスタ {cluster_id}:")
            print(f"    target_amount_tableau平均: {target_stats['平均']:.2f}")
            print(f"    target_amount_tableau中央値: {target_stats['中央値']:.2f}")
            print(f"    人流平均: {feature_stats['人流平均']:.0f}")
            print(f"    席数平均: {feature_stats['席数平均']:.0f}")
    
    # 結果を保存
    results[cuisine] = {
        'data': cuisine_data,
        'silhouette_score': silhouette_avg,
        'cluster_stats': cluster_stats,
        'cluster_counts': cluster_counts
    }
    
    # 全体の統計に追加
    cluster_stats_all[cuisine] = cluster_stats

# 結果をCSVファイルに保存
print(f"\n=== 結果をCSVファイルに保存 ===")

# 各料理カテゴリのクラスタ統計をまとめて保存
all_stats_df = pd.DataFrame()

for cuisine, stats in cluster_stats_all.items():
    for cluster_id, cluster_stat in stats.items():
        col_name = f"{cuisine}_クラスタ{cluster_id}"
        all_stats_df[col_name] = cluster_stat

all_stats_df.to_csv('cuisine_cluster_stats.csv', encoding='utf-8-sig')
print(f"料理カテゴリ別クラスタ統計を 'cuisine_cluster_stats.csv' に保存しました。")

# シルエットスコアの比較
print(f"\n=== シルエットスコア比較 ===")
silhouette_scores = {cuisine: results[cuisine]['silhouette_score'] for cuisine in results.keys()}
silhouette_df = pd.DataFrame(list(silhouette_scores.items()), columns=['料理カテゴリ', 'シルエットスコア'])
silhouette_df = silhouette_df.sort_values('シルエットスコア', ascending=False)

print(silhouette_df.round(3))

# シルエットスコアをCSVに保存
silhouette_df.to_csv('cuisine_silhouette_scores.csv', encoding='utf-8-sig', index=False)
print(f"シルエットスコア比較を 'cuisine_silhouette_scores.csv' に保存しました。")

# 可視化
print(f"\n=== 可視化 ===")

# 上位5つの料理カテゴリのクラスタリング結果を可視化
top_cuisines = silhouette_df.head(5)['料理カテゴリ'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, cuisine in enumerate(top_cuisines):
    if i >= 5:
        break
        
    cuisine_data = results[cuisine]['data']
    
    # 散布図を作成
    scatter = axes[i].scatter(
        cuisine_data['AVG_MONTHLY_POPULATION'], 
        cuisine_data['NUM_SEATS'], 
        c=cuisine_data['cluster'], 
        cmap='viridis', 
        alpha=0.6
    )
    
    axes[i].set_xlabel('AVG_MONTHLY_POPULATION (人流)')
    axes[i].set_ylabel('NUM_SEATS (席数)')
    axes[i].set_title(f'{cuisine}\n(シルエットスコア: {results[cuisine]["silhouette_score"]:.3f})')
    axes[i].grid(True, alpha=0.3)

# 最後のサブプロットを削除
if len(top_cuisines) < 6:
    fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('cuisine_clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

# シルエットスコアの比較グラフ
plt.figure(figsize=(12, 8))
plt.bar(range(len(silhouette_df)), silhouette_df['シルエットスコア'])
plt.xlabel('料理カテゴリ')
plt.ylabel('シルエットスコア')
plt.title('各料理カテゴリのシルエットスコア比較')
plt.xticks(range(len(silhouette_df)), silhouette_df['料理カテゴリ'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cuisine_silhouette_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 各料理カテゴリのクラスタ分布を可視化
fig, ax = plt.subplots(figsize=(15, 8))

# クラスタ分布データを準備
cluster_dist_data = []
for cuisine in valid_cuisines:
    if cuisine in results:
        for cluster_id in range(3):
            count = results[cuisine]['cluster_counts'].get(cluster_id, 0)
            cluster_dist_data.append({
                '料理カテゴリ': cuisine,
                'クラスタ': f'クラスタ{cluster_id}',
                '店舗数': count
            })

cluster_dist_df = pd.DataFrame(cluster_dist_data)
cluster_dist_pivot = cluster_dist_df.pivot(index='料理カテゴリ', columns='クラスタ', values='店舗数').fillna(0)

cluster_dist_pivot.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('各料理カテゴリのクラスタ分布')
ax.set_xlabel('料理カテゴリ')
ax.set_ylabel('店舗数')
ax.legend(title='クラスタ', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cuisine_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n分析完了！結果は以下のファイルに保存されました：")
print(f"- cuisine_cluster_stats.csv: 料理カテゴリ別クラスタ統計")
print(f"- cuisine_silhouette_scores.csv: シルエットスコア比較")
print(f"- cuisine_clustering_results.png: 上位5カテゴリのクラスタリング結果")
print(f"- cuisine_silhouette_comparison.png: シルエットスコア比較グラフ")
print(f"- cuisine_cluster_distribution.png: クラスタ分布")
