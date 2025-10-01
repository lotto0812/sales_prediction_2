import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
print("データを読み込み中...")
df = pd.read_excel('aggregated_df.xlsx')

# 分析用データの作成（20 < target_amount_tableau < 1000の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 1000)
df = df[analysis_condition].copy()

print(f"分析対象データ形状: {df.shape}")

# target_amount_tableauの範囲を定義
def create_amount_ranges(df):
    """target_amount_tableauの範囲を作成"""
    df['amount_range'] = pd.cut(
        df['target_amount_tableau'],
        bins=[20, 75, 150, 300, 1000],
        labels=['20~75', '75~150', '150~300', '300~'],
        right=False
    )
    return df

# 範囲を作成
df_with_ranges = create_amount_ranges(df)

# 各ランクでの料理カテゴリ別データ数を集計
ranges = ['20~75', '75~150', '150~300', '300~']

print("\n=== 各ランクでの料理カテゴリ別データ数Top10 ===")

for range_name in ranges:
    print(f"\n【{range_name}ランク】")
    
    # 該当ランクのデータを抽出
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    
    if len(df_range) == 0:
        print("データなし")
        continue
    
    # 料理カテゴリ別のデータ数を集計
    cuisine_counts = df_range['CUISINE_CAT_origin'].value_counts()
    
    # Top10を表示
    top10 = cuisine_counts.head(10)
    
    print(f"総データ数: {len(df_range):,}店舗")
    print(f"料理カテゴリ数: {len(cuisine_counts)}カテゴリ")
    print("\nTop10料理カテゴリ:")
    print("-" * 50)
    
    for i, (cuisine, count) in enumerate(top10.items(), 1):
        percentage = count / len(df_range) * 100
        print(f"{i:2d}. {cuisine:<20} {count:4,}店舗 ({percentage:5.1f}%)")
    
    # 上位10カテゴリの累積割合
    cumulative_percentage = top10.sum() / len(df_range) * 100
    print(f"\n上位10カテゴリの累積割合: {cumulative_percentage:.1f}%")

# 全体での料理カテゴリ別データ数も表示
print(f"\n【全体】")
cuisine_counts_all = df_with_ranges['CUISINE_CAT_origin'].value_counts()
top10_all = cuisine_counts_all.head(10)

print(f"総データ数: {len(df_with_ranges):,}店舗")
print(f"料理カテゴリ数: {len(cuisine_counts_all)}カテゴリ")
print("\nTop10料理カテゴリ:")
print("-" * 50)

for i, (cuisine, count) in enumerate(top10_all.items(), 1):
    percentage = count / len(df_with_ranges) * 100
    print(f"{i:2d}. {cuisine:<20} {count:4,}店舗 ({percentage:5.1f}%)")

# 各ランクの料理カテゴリ分布を可視化
print(f"\n=== 可視化 ===")

# 各ランクのTop10料理カテゴリを棒グラフで表示
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, range_name in enumerate(ranges):
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    
    if len(df_range) > 0:
        cuisine_counts = df_range['CUISINE_CAT_origin'].value_counts()
        top10 = cuisine_counts.head(10)
        
        # 棒グラフを作成
        bars = axes[i].bar(range(len(top10)), top10.values, color='skyblue', alpha=0.7)
        axes[i].set_title(f'{range_name}ランク - Top10料理カテゴリ\n(総データ数: {len(df_range):,}店舗)', 
                         fontsize=14, pad=20)
        axes[i].set_xlabel('料理カテゴリ', fontsize=12)
        axes[i].set_ylabel('店舗数', fontsize=12)
        axes[i].set_xticks(range(len(top10)))
        axes[i].set_xticklabels(top10.index, rotation=45, ha='right')
        axes[i].grid(axis='y', alpha=0.3)
        
        # 各バーの上に数値を表示
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    else:
        axes[i].text(0.5, 0.5, f'データなし\n({range_name})', 
                    ha='center', va='center', transform=axes[i].transAxes, fontsize=14)
        axes[i].set_title(f'{range_name}ランク', fontsize=14)

plt.tight_layout()
plt.savefig('cuisine_rank_distribution.png', dpi=300, bbox_inches='tight')
plt.close()  # メモリ節約のため
print("料理カテゴリ分布グラフを 'cuisine_rank_distribution.png' に保存しました。")

# 各ランクの料理カテゴリ分布をヒートマップで表示
print(f"\n=== 料理カテゴリ×ランク ヒートマップ ===")

# 上位20料理カテゴリを選択
top20_cuisines = cuisine_counts_all.head(20).index

# ヒートマップ用のデータを作成
heatmap_data = []
for cuisine in top20_cuisines:
    row = []
    for range_name in ranges:
        count = len(df_with_ranges[(df_with_ranges['CUISINE_CAT_origin'] == cuisine) & 
                                  (df_with_ranges['amount_range'] == range_name)])
        row.append(count)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, 
                         index=top20_cuisines, 
                         columns=ranges)

plt.figure(figsize=(12, 16))
sns.heatmap(heatmap_df, 
            annot=True, 
            fmt='d', 
            cmap='YlOrRd', 
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
plt.title('料理カテゴリ×ランク データ数分布\n(上位20料理カテゴリ)', fontsize=16, pad=20)
plt.xlabel('target_amount_tableauの範囲', fontsize=12)
plt.ylabel('料理カテゴリ', fontsize=12)
plt.tight_layout()
plt.savefig('cuisine_rank_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()  # メモリ節約のため
print("料理カテゴリ×ランク ヒートマップを 'cuisine_rank_heatmap.png' に保存しました。")

# 各ランクの料理カテゴリ分布をCSVに保存
print(f"\n=== 結果をCSVファイルに保存 ===")

# 各ランクのTop10をCSVに保存
for range_name in ranges:
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    if len(df_range) > 0:
        cuisine_counts = df_range['CUISINE_CAT_origin'].value_counts()
        top10 = cuisine_counts.head(10)
        
        # DataFrameに変換
        top10_df = pd.DataFrame({
            '料理カテゴリ': top10.index,
            '店舗数': top10.values,
            '割合(%)': (top10.values / len(df_range) * 100).round(1)
        })
        
        filename = f'top10_cuisines_{range_name.replace("~", "_")}.csv'
        top10_df.to_csv(filename, encoding='utf-8-sig', index=False)
        print(f"{range_name}ランクのTop10を '{filename}' に保存しました。")

# 全体のTop20をCSVに保存
top20_df = pd.DataFrame({
    '料理カテゴリ': cuisine_counts_all.head(20).index,
    '店舗数': cuisine_counts_all.head(20).values,
    '割合(%)': (cuisine_counts_all.head(20).values / len(df_with_ranges) * 100).round(1)
})
top20_df.to_csv('top20_cuisines_overall.csv', encoding='utf-8-sig', index=False)
print(f"全体のTop20を 'top20_cuisines_overall.csv' に保存しました。")

# ヒートマップデータもCSVに保存
heatmap_df.to_csv('cuisine_rank_heatmap_data.csv', encoding='utf-8-sig')
print(f"ヒートマップデータを 'cuisine_rank_heatmap_data.csv' に保存しました。")

print(f"\n分析完了！")

