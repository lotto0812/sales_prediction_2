import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import japanize_matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

# 結果保存用フォルダを作成
results_folder = 'cluster_analysis_results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"結果保存フォルダを作成しました: {results_folder}")

# データの読み込み
print("データを読み込み中...")
df = pd.read_csv('aggregated_df_with_cuisine_clusters.csv')

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

print(f"\n各範囲のデータ数:")
print(df['amount_range'].value_counts().sort_index())

# 分析用データの作成（20 < target_amount_tableau < 600の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 600)
df_analysis = df[analysis_condition].copy()
excluded_count = (~analysis_condition).sum()

print(f"\n分析対象データ形状: {df_analysis.shape}")
print(f"除外率: {excluded_count / len(df) * 100:.1f}%")

print("\n列名:")
print(df.columns.tolist())

print("\n最初の5行:")
print(df.head())

print("\nデータの基本情報:")
print(df.info())

print("\n欠損値の確認:")
print(df.isnull().sum())

# 相関係数の計算
print("\n=== 相関係数の計算 ===")
# target_amount_tableauとの相関を計算（分析用データを使用）
correlation_vars = ['target_amount_tableau', 'AVG_MONTHLY_POPULATION', 'NUM_SEATS', 'NEAREST_STATION_INFO_count', 'DINNER_PRICE', 'IS_FAMILY_FRIENDLY', 'cuisine_cluster']
correlation_df = df_analysis[correlation_vars].corr()

print("\ntarget_amount_tableauとの相関係数:")
print(correlation_df['target_amount_tableau'].sort_values(ascending=False))

# 散布図で可視化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 人流 vs target_amount_tableau
axes[0,0].scatter(df_analysis['AVG_MONTHLY_POPULATION'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[0,0].set_xlabel('AVG_MONTHLY_POPULATION (人流)')
axes[0,0].set_ylabel('target_amount_tableau')
axes[0,0].set_title(f'人流 vs target_amount_tableau\n相関係数: {correlation_df.loc["AVG_MONTHLY_POPULATION", "target_amount_tableau"]:.3f}')

# 席数 vs target_amount_tableau
axes[0,1].scatter(df_analysis['NUM_SEATS'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[0,1].set_xlabel('NUM_SEATS (席数)')
axes[0,1].set_ylabel('target_amount_tableau')
axes[0,1].set_title(f'席数 vs target_amount_tableau\n相関係数: {correlation_df.loc["NUM_SEATS", "target_amount_tableau"]:.3f}')

# 最寄り駅数 vs target_amount_tableau
axes[0,2].scatter(df_analysis['NEAREST_STATION_INFO_count'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[0,2].set_xlabel('NEAREST_STATION_INFO_count (最寄り駅数)')
axes[0,2].set_ylabel('target_amount_tableau')
axes[0,2].set_title(f'最寄り駅数 vs target_amount_tableau\n相関係数: {correlation_df.loc["NEAREST_STATION_INFO_count", "target_amount_tableau"]:.3f}')

# ディナー価格 vs target_amount_tableau
axes[1,0].scatter(df_analysis['DINNER_PRICE'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[1,0].set_xlabel('DINNER_PRICE (ディナー価格)')
axes[1,0].set_ylabel('target_amount_tableau')
axes[1,0].set_title(f'ディナー価格 vs target_amount_tableau\n相関係数: {correlation_df.loc["DINNER_PRICE", "target_amount_tableau"]:.3f}')

# ファミリー向けフラグ vs target_amount_tableau
axes[1,1].scatter(df_analysis['IS_FAMILY_FRIENDLY'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[1,1].set_xlabel('IS_FAMILY_FRIENDLY (ファミリー向け)')
axes[1,1].set_ylabel('target_amount_tableau')
axes[1,1].set_title(f'ファミリー向け vs target_amount_tableau\n相関係数: {correlation_df.loc["IS_FAMILY_FRIENDLY", "target_amount_tableau"]:.3f}')

# 料理クラスタ vs target_amount_tableau
axes[1,2].scatter(df_analysis['cuisine_cluster'], df_analysis['target_amount_tableau'], alpha=0.1)
axes[1,2].set_xlabel('cuisine_cluster (料理クラスタ)')
axes[1,2].set_ylabel('target_amount_tableau')
axes[1,2].set_title(f'料理クラスタ vs target_amount_tableau\n相関係数: {correlation_df.loc["cuisine_cluster", "target_amount_tableau"]:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n人流(AVG_MONTHLY_POPULATION)とtarget_amount_tableauの相関係数: {correlation_df.loc['AVG_MONTHLY_POPULATION', 'target_amount_tableau']:.3f}")
print(f"席数(NUM_SEATS)とtarget_amount_tableauの相関係数: {correlation_df.loc['NUM_SEATS', 'target_amount_tableau']:.3f}")
print(f"最寄り駅数(NEAREST_STATION_INFO_count)とtarget_amount_tableauの相関係数: {correlation_df.loc['NEAREST_STATION_INFO_count', 'target_amount_tableau']:.3f}")
print(f"ディナー価格(DINNER_PRICE)とtarget_amount_tableauの相関係数: {correlation_df.loc['DINNER_PRICE', 'target_amount_tableau']:.3f}")
print(f"ファミリー向け(IS_FAMILY_FRIENDLY)とtarget_amount_tableauの相関係数: {correlation_df.loc['IS_FAMILY_FRIENDLY', 'target_amount_tableau']:.3f}")
print(f"料理クラスタ(cuisine_cluster)とtarget_amount_tableauの相関係数: {correlation_df.loc['cuisine_cluster', 'target_amount_tableau']:.3f}")

# クラスタリングの実装
print("\n=== クラスタリングの実装 ===")

# クラスタリング用の特徴量を準備（分析用データを使用）
clustering_features = ['AVG_MONTHLY_POPULATION', 'NUM_SEATS', 'NEAREST_STATION_INFO_count', 'DINNER_PRICE', 'IS_FAMILY_FRIENDLY', 'cuisine_cluster']
X = df_analysis[clustering_features].copy()

print(f"クラスタリング用特徴量: {clustering_features}")
print(f"特徴量数: {len(clustering_features)}個")

# 欠損値の確認
print(f"欠損値の確認:")
print(X.isnull().sum())

# 欠損値を含む行を除外
X_clean = X.dropna()
print(f"\n欠損値除外後のデータ形状: {X_clean.shape}")
print(f"除外されたデータ数: {len(X) - len(X_clean)}")

# 分析用データも同様に欠損値を除外
df_analysis_clean = df_analysis.dropna(subset=clustering_features)
print(f"分析用データ（欠損値除外後）: {df_analysis_clean.shape}")

# クリーンなデータを使用
X = X_clean
df_analysis = df_analysis_clean

# データの基本統計
print(f"\nクラスタリング用特徴量の基本統計:")
print(X.describe())

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 最適なクラスタ数を決定（エルボー法とシルエット分析）
print("\n最適なクラスタ数の決定中...")

# エルボー法
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# エルボー法の可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True)

# シルエット分析の可視化
ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'clustering_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 最適なクラスタ数を選択（シルエットスコアが最も高いk）
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"最適なクラスタ数: {optimal_k} (シルエットスコア: {max(silhouette_scores):.3f})")

# 最終的なクラスタリング
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# クラスタラベルをデータフレームに追加（分析用データに追加）
df_analysis['cluster'] = cluster_labels

# クラスタリング結果の可視化（2つの主要な特徴量で表示）
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X['AVG_MONTHLY_POPULATION'], X['NUM_SEATS'], 
                     c=cluster_labels, cmap='viridis', alpha=0.1)
plt.xlabel('AVG_MONTHLY_POPULATION (人流)')
plt.ylabel('NUM_SEATS (席数)')
plt.title(f'クラスタリング結果 (k={optimal_k})\n特徴量: 人流, 席数, 最寄り駅数, ディナー価格, ファミリー向け, 料理クラスタ')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_folder, 'clustering_result.png'), dpi=300, bbox_inches='tight')
plt.close()

# 各クラスタの基本情報
print(f"\n各クラスタの基本情報:")
cluster_info = df_analysis.groupby('cluster')[clustering_features].agg(['count', 'mean', 'std', 'min', 'max'])
print(cluster_info)

print(f"\n各クラスタのデータ数:")
print(df_analysis['cluster'].value_counts().sort_index())

# 各クラスタのtarget_amount_tableauの詳細統計（範囲別）
print("\n=== 各クラスタのtarget_amount_tableau統計情報（範囲別） ===")

# 統計情報を計算
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

# 各クラスタの統計情報を計算（範囲別）
cluster_range_stats = {}
for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    cluster_range_stats[cluster_id] = {}
    
    # 各範囲での統計を計算
    for range_name in sorted(df_analysis['amount_range'].unique()):
        range_data = cluster_data[cluster_data['amount_range'] == range_name]['target_amount_tableau']
        if len(range_data) > 0:
            cluster_range_stats[cluster_id][range_name] = calculate_percentile_stats(range_data)

# 結果を整理して表示
print("\n各クラスタ・範囲別のtarget_amount_tableau統計情報:")
for cluster_id in sorted(cluster_range_stats.keys()):
    print(f"\n【クラスタ {cluster_id}】")
    for range_name in sorted(cluster_range_stats[cluster_id].keys()):
        stats = cluster_range_stats[cluster_id][range_name]
        print(f"  {range_name}: データ数={stats['データ数']:,}, 平均={stats['平均']:.2f}, 中央値={stats['中央値']:.2f}")

# 全体の範囲別統計も表示
print(f"\n=== 全体の範囲別統計 ===")
overall_range_stats = {}
for range_name in sorted(df['amount_range'].unique()):
    range_data = df[df['amount_range'] == range_name]['target_amount_tableau']
    if len(range_data) > 0:
        overall_range_stats[range_name] = calculate_percentile_stats(range_data)
        stats = overall_range_stats[range_name]
        print(f"{range_name}: データ数={stats['データ数']:,}, 平均={stats['平均']:.2f}, 中央値={stats['中央値']:.2f}")

# 統計情報をCSVファイルに保存
# クラスタ別統計
cluster_stats_df = pd.DataFrame()
for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]['target_amount_tableau']
    cluster_stats_df[f'クラスタ_{cluster_id}'] = calculate_percentile_stats(cluster_data)

cluster_stats_df.to_csv(os.path.join(results_folder, 'cluster_target_amount_stats.csv'), encoding='utf-8-sig')
print(f"\nクラスタ統計情報を '{results_folder}/cluster_target_amount_stats.csv' に保存しました。")

# 範囲別統計
range_stats_df = pd.DataFrame(overall_range_stats).T
range_stats_df.to_csv(os.path.join(results_folder, 'range_target_amount_stats.csv'), encoding='utf-8-sig')
print(f"範囲別統計情報を '{results_folder}/range_target_amount_stats.csv' に保存しました。")

# 各クラスタの分布を可視化（範囲別）
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 箱ひげ図（範囲別）
df_analysis.boxplot(column='target_amount_tableau', by='amount_range', ax=axes[0,0])
axes[0,0].set_title('各範囲のtarget_amount_tableau分布（箱ひげ図）')
axes[0,0].set_xlabel('target_amount_tableauの範囲')
axes[0,0].set_ylabel('target_amount_tableau')

# 2. ヒストグラム（範囲別、重ね合わせ）
for range_name in sorted(df_analysis['amount_range'].unique()):
    range_data = df_analysis[df_analysis['amount_range'] == range_name]['target_amount_tableau']
    if len(range_data) > 0:
        axes[0,1].hist(range_data, alpha=0.6, label=f'{range_name}', bins=30)
axes[0,1].set_title('各範囲のtarget_amount_tableau分布（ヒストグラム）')
axes[0,1].set_xlabel('target_amount_tableau')
axes[0,1].set_ylabel('頻度')
axes[0,1].legend()

# 3. クラスタ別の散布図（人流 vs target_amount_tableau）
for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    axes[1,0].scatter(cluster_data['AVG_MONTHLY_POPULATION'], 
                     cluster_data['target_amount_tableau'], 
                     alpha=0.1, label=f'クラスタ {cluster_id}')
axes[1,0].set_xlabel('AVG_MONTHLY_POPULATION (人流)')
axes[1,0].set_ylabel('target_amount_tableau')
axes[1,0].set_title('人流 vs target_amount_tableau（クラスタ別）')
axes[1,0].legend()

# 4. クラスタ別の散布図（席数 vs target_amount_tableau）
for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    axes[1,1].scatter(cluster_data['NUM_SEATS'], 
                     cluster_data['target_amount_tableau'], 
                     alpha=0.1, label=f'クラスタ {cluster_id}')
axes[1,1].set_xlabel('NUM_SEATS (席数)')
axes[1,1].set_ylabel('target_amount_tableau')
axes[1,1].set_title('席数 vs target_amount_tableau（クラスタ別）')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'cluster_target_amount_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 範囲別のクラスタ分布を可視化
fig, ax = plt.subplots(figsize=(12, 8))
cluster_range_counts = df_analysis.groupby(['amount_range', 'cluster']).size().unstack(fill_value=0)
cluster_range_counts.plot(kind='bar', ax=ax, stacked=True)
ax.set_title('各範囲でのクラスタ分布')
ax.set_xlabel('target_amount_tableauの範囲')
ax.set_ylabel('店舗数')
ax.legend(title='クラスタ', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'cluster_distribution_by_range.png'), dpi=300, bbox_inches='tight')
plt.close()

# クラスタ毎のtarget_amount_tableauのヒストグラム
print("\n=== クラスタ毎のtarget_amount_tableauヒストグラム作成中 ===")
plt.figure(figsize=(15, 10))

# 各クラスタのヒストグラムを重ね合わせて表示
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, cluster_id in enumerate(sorted(df_analysis['cluster'].unique())):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]['target_amount_tableau']
    plt.hist(cluster_data, bins=50, alpha=0.1, color=colors[i], 
             label=f'クラスタ {cluster_id} (n={len(cluster_data):,})', density=True)

plt.xlabel('target_amount_tableau')
plt.ylabel('密度')
plt.title('クラスタ毎のtarget_amount_tableau分布（ヒストグラム）\nα=0.1で重ね合わせ表示')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_folder, 'cluster_histogram_overlay.png'), dpi=300, bbox_inches='tight')
plt.close()

# 各クラスタを個別のサブプロットで表示
n_clusters = len(df_analysis['cluster'].unique())
n_cols = 3
n_rows = (n_clusters + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, cluster_id in enumerate(sorted(df_analysis['cluster'].unique())):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]['target_amount_tableau']
    
    axes[i].hist(cluster_data, bins=30, alpha=0.1, color=colors[i], edgecolor='black')
    axes[i].set_xlabel('target_amount_tableau')
    axes[i].set_ylabel('頻度')
    axes[i].set_title(f'クラスタ {cluster_id}\n(n={len(cluster_data):,}, 平均={cluster_data.mean():.1f})')
    axes[i].grid(True, alpha=0.3)

# 余分なサブプロットは非表示
for i in range(n_clusters, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'cluster_histogram_individual.png'), dpi=300, bbox_inches='tight')
plt.close()

# クラスタの特徴をまとめる
print("\n=== 各クラスタの特徴まとめ ===")
for cluster_id in sorted(df_analysis['cluster'].unique()):
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
    print(f"\n【クラスタ {cluster_id}】")
    print(f"  データ数: {len(cluster_data):,}店舗")
    print(f"  人流の平均: {cluster_data['AVG_MONTHLY_POPULATION'].mean():.0f}")
    print(f"  席数の平均: {cluster_data['NUM_SEATS'].mean():.0f}")
    print(f"  最寄り駅数の平均: {cluster_data['NEAREST_STATION_INFO_count'].mean():.1f}")
    print(f"  ディナー価格の平均: {cluster_data['DINNER_PRICE'].mean():.0f}")
    print(f"  ファミリー向けの平均: {cluster_data['IS_FAMILY_FRIENDLY'].mean():.2f}")
    print(f"  料理クラスタの平均: {cluster_data['cuisine_cluster'].mean():.2f}")
    print(f"  target_amount_tableauの平均: {cluster_data['target_amount_tableau'].mean():.2f}")
    print(f"  target_amount_tableauの中央値: {cluster_data['target_amount_tableau'].median():.2f}")

# 除外されたデータの情報も表示
print(f"\n=== 除外されたデータの情報 ===")
excluded_data = df[~analysis_condition]
print(f"除外されたデータ数: {len(excluded_data):,}店舗")
print(f"除外されたデータのtarget_amount_tableau統計:")
excluded_stats = calculate_percentile_stats(excluded_data['target_amount_tableau'])
for key, value in excluded_stats.items():
    print(f"  {key}: {value:.2f}")

print(f"\n分析完了！結果は以下のファイルに保存されました：")
print(f"- {results_folder}/cluster_target_amount_stats.csv: 各クラスタの統計情報")
print(f"- {results_folder}/range_target_amount_stats.csv: 範囲別統計情報")
print(f"- {results_folder}/correlation_analysis.png: 相関分析の可視化")
print(f"- {results_folder}/clustering_analysis.png: クラスタ数決定の分析")
print(f"- {results_folder}/clustering_result.png: クラスタリング結果")
print(f"- {results_folder}/cluster_target_amount_analysis.png: クラスタ別の詳細分析")
print(f"- {results_folder}/cluster_distribution_by_range.png: 範囲別クラスタ分布")
print(f"- {results_folder}/cluster_histogram_overlay.png: クラスタ別ヒストグラム（重ね合わせ）")
print(f"- {results_folder}/cluster_histogram_individual.png: クラスタ別ヒストグラム（個別）")
