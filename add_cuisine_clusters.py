import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('aggregated_df_with_predictions.csv')

analysis_condition = (df['target_amount_tableau'] > 20) & (df['target_amount_tableau'] < 600)
df_analysis = df[analysis_condition].copy()

cuisine_stats = df_analysis.groupby('CUISINE_CAT_origin')['target_amount_tableau'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).reset_index()

cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 30]

X = cuisine_stats[['mean', 'std', 'min', 'max', 'median']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

cuisine_stats['cuisine_cluster'] = cluster_labels

cuisine_cluster_mapping = dict(zip(cuisine_stats['CUISINE_CAT_origin'], cuisine_stats['cuisine_cluster']))

df['cuisine_cluster'] = df['CUISINE_CAT_origin'].map(cuisine_cluster_mapping)

df.to_csv('aggregated_df_with_cuisine_clusters.csv', encoding='utf-8-sig', index=False)

print(f"cuisine_cluster列を追加して 'aggregated_df_with_cuisine_clusters.csv' に保存しました。")
print(f"cuisine_clusterの分布:")
print(df['cuisine_cluster'].value_counts().sort_index())
