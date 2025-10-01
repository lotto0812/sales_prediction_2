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

print(f"元データ形状: {df.shape}")

# 分析用データの作成（20 < target_amount_tableau < 1000の範囲のみ）
analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 1000)
df = df[analysis_condition].copy()

print(f"分析対象データ形状: {df.shape}")
print(f"除外率: {(1 - len(df) / 23535) * 100:.1f}%")

# 使用する変数の定義
variables = [
    'target_amount_tableau',
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
    'rate_count',
    'seats_rate_count'
]

# target_amount_tableauの範囲を定義（20 < target_amount_tableau < 1000の範囲内で）
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

# 各範囲のデータ数を確認
print("\n各範囲のデータ数:")
range_counts = df_with_ranges['amount_range'].value_counts().sort_index()
print(range_counts)

# 各ランクの詳細統計情報を表示
print("\n=== 各ランクの詳細統計情報 ===")
for range_name in ['20~75', '75~150', '150~300', '300~']:
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    if len(df_range) > 0:
        print(f"\n【{range_name}ランク】")
        print(f"  データ数: {len(df_range):,}店舗")
        print(f"  割合: {len(df_range)/len(df_with_ranges)*100:.1f}%")
        print(f"  target_amount_tableauの範囲: {df_range['target_amount_tableau'].min():.2f} - {df_range['target_amount_tableau'].max():.2f}")
        print(f"  平均: {df_range['target_amount_tableau'].mean():.2f}")
        print(f"  中央値: {df_range['target_amount_tableau'].median():.2f}")
        print(f"  標準偏差: {df_range['target_amount_tableau'].std():.2f}")
        print(f"  25%タイル: {df_range['target_amount_tableau'].quantile(0.25):.2f}")
        print(f"  75%タイル: {df_range['target_amount_tableau'].quantile(0.75):.2f}")
        
        # 主要変数の平均値も表示
        print(f"  人流平均: {df_range['AVG_MONTHLY_POPULATION'].mean():.0f}")
        print(f"  席数平均: {df_range['NUM_SEATS'].mean():.1f}")
        print(f"  評価数平均: {df_range['RATING_CNT'].mean():.1f}")
        print(f"  評価スコア平均: {df_range['RATING_SCORE'].mean():.2f}")
    else:
        print(f"\n【{range_name}ランク】")
        print(f"  データ数: 0店舗")

# 相関ヒートマップを作成する関数
def create_correlation_heatmap(df_subset, title, ax):
    """相関ヒートマップを作成"""
    # 数値変数のみを選択
    numeric_vars = df_subset[variables].select_dtypes(include=[np.number])
    
    # 相関行列を計算
    corr_matrix = numeric_vars.corr()
    
    # ヒートマップを作成
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 4}  # アノテーションのフォントサイズを小さく
    )
    
    ax.set_title(title, fontsize=12, pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    # X軸ラベルを右寄せに設定
    for label in ax.get_xticklabels():
        label.set_ha('right')

# 各範囲の相関ヒートマップを作成
ranges = ['20~75', '75~150', '150~300', '300~']

# サブプロットの設定（2行2列）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, range_name in enumerate(ranges):
    # 該当範囲のデータを抽出
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    
    if len(df_range) > 0:
        # 相関ヒートマップを作成
        create_correlation_heatmap(
            df_range, 
            f'target_amount_tableau {range_name}\n(データ数: {len(df_range):,})', 
            axes[i]
        )
        
        # 各範囲の基本統計を表示
        print(f"\n=== {range_name} の基本統計 ===")
        print(f"データ数: {len(df_range):,}")
        if len(df_range) > 0:
            print(f"target_amount_tableauの範囲: {df_range['target_amount_tableau'].min():.2f} - {df_range['target_amount_tableau'].max():.2f}")
            print(f"平均: {df_range['target_amount_tableau'].mean():.2f}")
            print(f"中央値: {df_range['target_amount_tableau'].median():.2f}")
    else:
        axes[i].text(0.5, 0.5, f'データなし\n({range_name})', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'target_amount_tableau {range_name}\n(データ数: 0)')

plt.tight_layout()
plt.savefig('correlation_heatmap_by_range.png', dpi=300, bbox_inches='tight')
plt.close()  # メモリ節約のため
# plt.show()  # コメントアウト

# 全体の相関ヒートマップも作成
print("\n=== 全体の相関ヒートマップ ===")
plt.figure(figsize=(12, 10))
numeric_vars_all = df[variables].select_dtypes(include=[np.number])
corr_matrix_all = numeric_vars_all.corr()

sns.heatmap(
    corr_matrix_all,
    annot=True,
    cmap='RdBu_r',
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 8}  # 全体のヒートマップのアノテーションサイズ
)

plt.title('全体の相関ヒートマップ', fontsize=16, pad=20)
plt.tick_params(axis='x', rotation=45)
plt.tick_params(axis='y', rotation=0)

# X軸ラベルを右寄せに設定
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_ha('right')
plt.tight_layout()
plt.savefig('correlation_heatmap_overall.png', dpi=300, bbox_inches='tight')
plt.close()  # メモリ節約のため
# plt.show()  # コメントアウト

# 各範囲でのtarget_amount_tableauとの相関を比較
print("\n=== 各範囲でのtarget_amount_tableauとの相関比較 ===")
correlation_comparison = pd.DataFrame()

for range_name in ranges:
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    
    if len(df_range) > 10:  # データ数が少なすぎる場合はスキップ
        numeric_vars_range = df_range[variables].select_dtypes(include=[np.number])
        corr_with_target = numeric_vars_range.corr()['target_amount_tableau'].drop('target_amount_tableau')
        correlation_comparison[range_name] = corr_with_target

# 相関比較表を表示
print(correlation_comparison.round(3))

# 各ランクで相関の絶対値が高いtop5を抽出
print("\n=== 各ランクでの相関絶対値Top5 ===")
top5_correlations = {}

for range_name in ranges:
    if range_name in correlation_comparison.columns:
        # 相関の絶対値を計算してソート
        abs_corr = correlation_comparison[range_name].abs().sort_values(ascending=False)
        top5 = abs_corr.head(5)
        top5_correlations[range_name] = top5
        
        print(f"\n【{range_name}】")
        for i, (var, abs_corr_val) in enumerate(top5.items(), 1):
            actual_corr = correlation_comparison[range_name][var]
            print(f"{i}. {var}: {actual_corr:.3f} (絶対値: {abs_corr_val:.3f})")

# Top5相関をDataFrameにまとめる
top5_df = pd.DataFrame(top5_correlations).fillna(0)
print(f"\n=== 各ランクでの相関絶対値Top5一覧表 ===")
print(top5_df.round(3))

# Top5相関をCSVに保存
top5_df.to_csv('top5_correlations_by_range.csv', encoding='utf-8-sig')
print(f"\nTop5相関表を 'top5_correlations_by_range.csv' に保存しました。")

# 相関比較を可視化
plt.figure(figsize=(15, 10))
sns.heatmap(
    correlation_comparison.T,
    annot=True,
    cmap='RdBu_r',
    center=0,
    fmt='.3f',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 8}  # 相関比較ヒートマップのアノテーションサイズ
)

# タイトルに各ランクのデータ数を追加
title_with_counts = '各範囲でのtarget_amount_tableauとの相関比較\n'
for range_name in ranges:
    df_range = df_with_ranges[df_with_ranges['amount_range'] == range_name]
    count = len(df_range)
    title_with_counts += f'{range_name}: {count:,}店舗  '
plt.title(title_with_counts, fontsize=16, pad=20)

plt.xlabel('変数')
plt.ylabel('target_amount_tableauの範囲')
plt.tick_params(axis='x', rotation=45)

# X軸ラベルを右寄せに設定
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_ha('right')
plt.tight_layout()
plt.savefig('correlation_comparison_by_range.png', dpi=300, bbox_inches='tight')
plt.close()  # メモリ節約のため
# plt.show()  # コメントアウト

print(f"\n分析完了！結果は以下のファイルに保存されました：")
print(f"- correlation_heatmap_by_range.png: 範囲別相関ヒートマップ")
print(f"- correlation_heatmap_overall.png: 全体の相関ヒートマップ")
print(f"- correlation_comparison_by_range.png: 相関比較ヒートマップ")
print(f"- top5_correlations_by_range.csv: 各ランクでの相関絶対値Top5一覧表")
