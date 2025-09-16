# ============================
# 単純 Train/Test 分割での学習・評価
# ============================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from lightgbm import LGBMClassifier

# ---- しきい値: low[20,50), middle[50,150), high[150,∞) ----
def make_class_label(y: pd.Series) -> pd.Series:
    bins = [20, 50, 150, np.inf]    # 下端を含む, 上端は含まない
    labels = ["low", "middle", "high"]
    return pd.cut(y, bins=bins, labels=labels, right=False)

# ラベル作成 & y<20 等を除外
df = df.copy()
df["y_class"] = make_class_label(df["target_amount_tableau"]).astype("category")
df = df[df["y_class"].notna()].copy()

# 学習に使わない列を除外
drop_cols = ["KEY", "target_amount_tableau", "RST_TITLE", "menu"]
use_cols = [c for c in df.columns if c not in drop_cols + ["y_class"]]

# 特徴量/目的
X_all = df[use_cols].copy()
y_all = df["y_class"].cat.set_categories(["low", "middle", "high"])

# Train/Test 分割（層化）
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# object列 → category（学習カテゴリ水準にテストを合わせる）
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
train_cat_levels = {}
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    train_cat_levels[c] = X_train[c].cat.categories.tolist()
    X_test[c] = X_test[c].astype("category").cat.set_categories(train_cat_levels[c])

# クラス重み（頻度の逆数）
class_counts = y_train.value_counts()
class_weight = {cls: (len(y_train) / (len(class_counts) * cnt)) for cls, cnt in class_counts.items()}

# モデル学習
clf = LGBMClassifier(
    objective="multiclass",
    learning_rate=0.05,
    n_estimators=3000,
    num_leaves=127,
    min_data_in_leaf=10,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    random_state=42,
    class_weight=class_weight,
)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="multi_logloss", verbose=False)

# 予測（確率は classes_ の順 → ["low","middle","high"] に並べ替え）
ORDER = ["low", "middle", "high"]
idx_map = [list(clf.classes_).index(k) for k in ORDER]
proba_test = clf.predict_proba(X_test)[:, idx_map]
pred_idx = proba_test.argmax(axis=1)
y_pred = pd.Categorical(np.array(ORDER)[pred_idx], categories=ORDER)

# 評価
print(classification_report(y_test, y_pred, labels=ORDER, digits=3))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred, labels=ORDER))

# テストデータに予測カラムを付与
df_test_scored = df.loc[X_test.index].copy()
df_test_scored["p_low"] = proba_test[:, 0]
df_test_scored["p_middle"] = proba_test[:, 1]
df_test_scored["p_high"] = proba_test[:, 2]
df_test_scored["pred_class"] = y_pred
df_test_scored["pred_confidence"] = proba_test.max(axis=1)

# df_test_scored が、予測付きのテストデータフレームです。
