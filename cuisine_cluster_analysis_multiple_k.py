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

# 複数のクラスタ数で分析
k_values = [3, 4, 5, 6]
all_results = {}

print(f"\n=== 複数クラスタ数での分析 ===")

for k in k_values:
    print(f"\n【クラスタ数: {k}】")
    results = {}
    
    for cuisine in valid_cuisines:
        # 該当カテゴリのデータを抽出
        cuisine_data = df_analysis[df_analysis['CUISINE_CAT_origin'] == cuisine].copy()
        
        if len(cuisine_data) < 10:  # データ数が少なすぎる場合はスキップ
            continue
        
        # クラスタリング用の特徴量を準備
        clustering_features = ['AVG_MONTHLY_POPULATION', 'NUM_SEATS']
        X = cuisine_data[clustering_features].copy()
        
        # 欠損値の確認
        if X.isnull().sum().sum() > 0:
            continue
        
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # クラスタリング実行
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # シルエットスコアを計算
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # クラスタラベルをデータフレームに追加
        cuisine_data['cluster'] = cluster_labels
        
        # 各クラスタの基本情報
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        # 各クラスタの統計情報を計算
        cluster_stats = {}
        for cluster_id in range(k):
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
        
        # 結果を保存
        results[cuisine] = {
            'data': cuisine_data,
            'silhouette_score': silhouette_avg,
            'cluster_stats': cluster_stats,
            'cluster_counts': cluster_counts
        }
    
    all_results[k] = results

# シルエットスコアの比較
print(f"\n=== クラスタ数別シルエットスコア比較 ===")

# 各クラスタ数での平均シルエットスコアを計算
k_comparison = []
for k in k_values:
    scores = [results[cuisine]['silhouette_score'] for cuisine in all_results[k].keys()]
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    k_comparison.append({
        'クラスタ数': k,
        '平均シルエットスコア': avg_score,
        '最大シルエットスコア': max_score,
        '最小シルエットスコア': min_score,
        '分析カテゴリ数': len(scores)
    })
    
    print(f"k={k}: 平均={avg_score:.3f}, 最大={max_score:.3f}, 最小={min_score:.3f}")

k_comparison_df = pd.DataFrame(k_comparison)
print(f"\nクラスタ数別比較:")
print(k_comparison_df.round(3))

# 各料理カテゴリでのクラスタ数別シルエットスコア比較
print(f"\n=== 各料理カテゴリでのクラスタ数別シルエットスコア ===")

cuisine_k_scores = {}
for cuisine in valid_cuisines:
    if cuisine in all_results[3]:  # k=3で分析されているカテゴリのみ
        scores = []
        for k in k_values:
            if cuisine in all_results[k]:
                scores.append(all_results[k][cuisine]['silhouette_score'])
            else:
                scores.append(np.nan)
        cuisine_k_scores[cuisine] = scores

cuisine_k_df = pd.DataFrame(cuisine_k_scores, index=[f'k={k}' for k in k_values]).T
cuisine_k_df = cuisine_k_df.sort_values('k=3', ascending=False)

print(f"\n上位10カテゴリのクラスタ数別シルエットスコア:")
print(cuisine_k_df.head(10).round(3))

# 可視化
print(f"\n=== 可視化 ===")

# 1. クラスタ数別平均シルエットスコアの比較
plt.figure(figsize=(10, 6))
plt.plot(k_values, k_comparison_df['平均シルエットスコア'], 'bo-', linewidth=2, markersize=8)
plt.xlabel('クラスタ数 (k)')
plt.ylabel('平均シルエットスコア')
plt.title('クラスタ数別平均シルエットスコア比較')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
for i, (k, score) in enumerate(zip(k_values, k_comparison_df['平均シルエットスコア'])):
    plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig('k_comparison_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 各料理カテゴリのクラスタ数別シルエットスコア（ヒートマップ）
plt.figure(figsize=(12, 8))
sns.heatmap(cuisine_k_df.head(15), annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'シルエットスコア'})
plt.title('各料理カテゴリのクラスタ数別シルエットスコア（上位15カテゴリ）')
plt.xlabel('クラスタ数')
plt.ylabel('料理カテゴリ')
plt.tight_layout()
plt.savefig('cuisine_k_silhouette_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 最適クラスタ数の分布
optimal_k_per_cuisine = []
for cuisine in cuisine_k_df.index:
    scores = cuisine_k_df.loc[cuisine].values
    if not np.isnan(scores).all():
        optimal_k_idx = np.nanargmax(scores)
        optimal_k = k_values[optimal_k_idx]
        optimal_k_per_cuisine.append(optimal_k)

optimal_k_counts = pd.Series(optimal_k_per_cuisine).value_counts().sort_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(optimal_k_counts.index, optimal_k_counts.values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
plt.xlabel('最適クラスタ数')
plt.ylabel('料理カテゴリ数')
plt.title('各料理カテゴリの最適クラスタ数分布')
plt.xticks(k_values)
for bar, count in zip(bars, optimal_k_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(count), ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimal_k_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 上位5カテゴリのクラスタ数別シルエットスコア比較
top_5_cuisines = cuisine_k_df.head(5).index

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, cuisine in enumerate(top_5_cuisines):
    scores = cuisine_k_df.loc[cuisine].values
    axes[i].plot(k_values, scores, 'bo-', linewidth=2, markersize=6)
    axes[i].set_title(f'{cuisine}')
    axes[i].set_xlabel('クラスタ数')
    axes[i].set_ylabel('シルエットスコア')
    axes[i].set_xticks(k_values)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('top5_cuisines_k_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 結果をCSVファイルに保存
print(f"\n=== 結果をCSVファイルに保存 ===")

# クラスタ数別比較を保存
k_comparison_df.to_csv('k_comparison_silhouette.csv', encoding='utf-8-sig', index=False)
print(f"クラスタ数別比較を 'k_comparison_silhouette.csv' に保存しました。")

# 各料理カテゴリのクラスタ数別シルエットスコアを保存
cuisine_k_df.to_csv('cuisine_k_silhouette_scores.csv', encoding='utf-8-sig')
print(f"料理カテゴリ別クラスタ数比較を 'cuisine_k_silhouette_scores.csv' に保存しました。")

# 最適クラスタ数分布を保存
optimal_k_df = pd.DataFrame({
    'クラスタ数': optimal_k_counts.index,
    'カテゴリ数': optimal_k_counts.values
})
optimal_k_df.to_csv('optimal_k_distribution.csv', encoding='utf-8-sig', index=False)
print(f"最適クラスタ数分布を 'optimal_k_distribution.csv' に保存しました。")

# 各クラスタ数での詳細統計を保存
for k in k_values:
    # 各料理カテゴリのクラスタ統計をまとめて保存
    all_stats_df = pd.DataFrame()
    
    for cuisine, stats in all_results[k].items():
        for cluster_id, cluster_stat in stats['cluster_stats'].items():
            col_name = f"{cuisine}_クラスタ{cluster_id}"
            all_stats_df[col_name] = cluster_stat
    
    filename = f'cuisine_cluster_stats_k{k}.csv'
    all_stats_df.to_csv(filename, encoding='utf-8-sig')
    print(f"クラスタ数{k}の統計情報を '{filename}' に保存しました。")

print(f"\n分析完了！結果は以下のファイルに保存されました：")
print(f"- k_comparison_silhouette.csv: クラスタ数別比較")
print(f"- cuisine_k_silhouette_scores.csv: 料理カテゴリ別クラスタ数比較")
print(f"- optimal_k_distribution.csv: 最適クラスタ数分布")
print(f"- cuisine_cluster_stats_k3.csv: クラスタ数3の統計情報")
print(f"- cuisine_cluster_stats_k4.csv: クラスタ数4の統計情報")
print(f"- cuisine_cluster_stats_k5.csv: クラスタ数5の統計情報")
print(f"- cuisine_cluster_stats_k6.csv: クラスタ数6の統計情報")
print(f"- k_comparison_silhouette.png: クラスタ数別平均シルエットスコア比較")
print(f"- cuisine_k_silhouette_heatmap.png: 料理カテゴリ別クラスタ数比較ヒートマップ")
print(f"- optimal_k_distribution.png: 最適クラスタ数分布")
print(f"- top5_cuisines_k_comparison.png: 上位5カテゴリのクラスタ数別比較")

