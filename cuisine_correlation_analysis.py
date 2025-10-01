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

# CUISINE_CAT_originの分布を確認
print(f"\nCUISINE_CAT_originの分布:")
cuisine_counts = df_with_ranges['CUISINE_CAT_origin'].value_counts()
print(cuisine_counts)

# データ数が30以上のカテゴリのみを対象
min_count = 30
valid_cuisines = cuisine_counts[cuisine_counts >= min_count].index
print(f"\n分析対象の料理カテゴリ（{min_count}店舗以上）:")
print(f"対象カテゴリ数: {len(valid_cuisines)}")
print(f"対象データ数: {df_with_ranges[df_with_ranges['CUISINE_CAT_origin'].isin(valid_cuisines)].shape[0]}")

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
    
    ax.set_title(title, fontsize=10, pad=15)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    # X軸ラベルを右寄せに設定
    for label in ax.get_xticklabels():
        label.set_ha('right')

# 各料理カテゴリで分析を実行
ranges = ['20~75', '75~150', '150~300', '300~']
all_results = {}

print(f"\n=== 各料理カテゴリでの相関分析 ===")

for cuisine in valid_cuisines:
    print(f"\n【{cuisine}】")
    
    # 該当カテゴリのデータを抽出
    cuisine_data = df_with_ranges[df_with_ranges['CUISINE_CAT_origin'] == cuisine].copy()
    print(f"データ数: {len(cuisine_data):,}店舗")
    
    if len(cuisine_data) < 30:
        print("データ数が少なすぎるためスキップ")
        continue
    
    # 各ランクの詳細統計情報を表示
    print(f"\n=== {cuisine} の各ランク詳細統計 ===")
    for range_name in ranges:
        df_range = cuisine_data[cuisine_data['amount_range'] == range_name]
        if len(df_range) > 0:
            print(f"\n【{range_name}ランク】")
            print(f"  データ数: {len(df_range):,}店舗")
            print(f"  割合: {len(df_range)/len(cuisine_data)*100:.1f}%")
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
    
    # 各範囲でのtarget_amount_tableauとの相関を計算
    print(f"\n=== {cuisine} の各ランクでの相関比較 ===")
    correlation_comparison = pd.DataFrame()
    
    for range_name in ranges:
        df_range = cuisine_data[cuisine_data['amount_range'] == range_name]
        
        if len(df_range) > 10:  # データ数が少なすぎる場合はスキップ
            numeric_vars_range = df_range[variables].select_dtypes(include=[np.number])
            corr_with_target = numeric_vars_range.corr()['target_amount_tableau'].drop('target_amount_tableau')
            correlation_comparison[range_name] = corr_with_target
    
    if not correlation_comparison.empty:
        # 相関比較表を表示
        print(correlation_comparison.round(3))
        
        # 各ランクで相関の絶対値が高いtop5を抽出
        print(f"\n=== {cuisine} の各ランクでの相関絶対値Top5 ===")
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
        print(f"\n=== {cuisine} の各ランクでの相関絶対値Top5一覧表 ===")
        print(top5_df.round(3))
        
        # 結果を保存
        all_results[cuisine] = {
            'correlation_comparison': correlation_comparison,
            'top5_correlations': top5_df,
            'data': cuisine_data
        }
        
        # 各料理カテゴリの相関ヒートマップを作成
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, range_name in enumerate(ranges):
            df_range = cuisine_data[cuisine_data['amount_range'] == range_name]
            
            if len(df_range) > 0:
                # 相関ヒートマップを作成
                create_correlation_heatmap(
                    df_range, 
                    f'{cuisine} - {range_name}\n(データ数: {len(df_range):,})', 
                    axes[i]
                )
            else:
                axes[i].text(0.5, 0.5, f'データなし\n({range_name})', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{cuisine} - {range_name}\n(データ数: 0)')
        
        plt.tight_layout()
        filename = f'correlation_heatmap_{cuisine.replace("/", "_").replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # メモリ節約のため
        # plt.show()  # コメントアウト
        print(f"相関ヒートマップを '{filename}' に保存しました。")
        
        # 相関比較を可視化
        if not correlation_comparison.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                correlation_comparison.T,
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.3f',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8}
            )
            
            # タイトルに各ランクのデータ数を追加
            title_with_counts = f'{cuisine} - 各ランクでのtarget_amount_tableauとの相関比較\n'
            for range_name in ranges:
                df_range = cuisine_data[cuisine_data['amount_range'] == range_name]
                count = len(df_range)
                title_with_counts += f'{range_name}: {count:,}店舗  '
            plt.title(title_with_counts, fontsize=14, pad=20)
            
            plt.xlabel('変数')
            plt.ylabel('target_amount_tableauの範囲')
            plt.tick_params(axis='x', rotation=45)
            
            # X軸ラベルを右寄せに設定
            ax = plt.gca()
            for label in ax.get_xticklabels():
                label.set_ha('right')
            plt.tight_layout()
            
            filename = f'correlation_comparison_{cuisine.replace("/", "_").replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()  # メモリ節約のため
            # plt.show()  # コメントアウト
            print(f"相関比較ヒートマップを '{filename}' に保存しました。")
        
        # CSVファイルに保存
        top5_filename = f'top5_correlations_{cuisine.replace("/", "_").replace(" ", "_")}.csv'
        top5_df.to_csv(top5_filename, encoding='utf-8-sig')
        print(f"Top5相関表を '{top5_filename}' に保存しました。")
        
        correlation_filename = f'correlation_comparison_{cuisine.replace("/", "_").replace(" ", "_")}.csv'
        correlation_comparison.to_csv(correlation_filename, encoding='utf-8-sig')
        print(f"相関比較表を '{correlation_filename}' に保存しました。")
    
    else:
        print(f"{cuisine} では有効な相関データがありません。")

# 全体のサマリーを作成
print(f"\n=== 全体サマリー ===")
print(f"分析対象料理カテゴリ数: {len(all_results)}")
print(f"生成されたファイル数: {len(all_results) * 4}")  # ヒートマップ、比較図、Top5、相関比較の各CSV

# 各料理カテゴリの主要相関をまとめる
summary_data = []
for cuisine, results in all_results.items():
    if 'top5_correlations' in results:
        top5_df = results['top5_correlations']
        for range_name in ranges:
            if range_name in top5_df.columns:
                top_vars = top5_df[range_name].nlargest(3).index.tolist()
                summary_data.append({
                    '料理カテゴリ': cuisine,
                    'ランク': range_name,
                    'Top3変数': ', '.join(top_vars)
                })

summary_df = pd.DataFrame(summary_data)
if not summary_df.empty:
    summary_df.to_csv('cuisine_correlation_summary.csv', encoding='utf-8-sig', index=False)
    print(f"全体サマリーを 'cuisine_correlation_summary.csv' に保存しました。")

print(f"\n分析完了！各料理カテゴリの相関分析結果が保存されました。")
