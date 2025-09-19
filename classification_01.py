# =========================================
# 1) ラベル作成（low/middle/high）と前処理
# =========================================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier

# ---- しきい値: low[20,50), middle[50,150), high[150,∞) ----
def make_class_label(y: pd.Series) -> pd.Series:
    bins = [20, 50, 150, np.inf]   # 区間下端を含む、上端は含まない
    labels = ["low", "middle", "high"]
    # y<20 は通常データに無い想定ですが、もしあれば NaN になります
    y_class = pd.cut(y, bins=bins, labels=labels, right=False)
    return y_class

# df: あなたの学習データ（下記の列を含む DataFrame）
# 例）df = pd.read_csv("train.csv")
# ここでは target_amount_tableau を教師にしてクラスを作る
df["y_class"] = make_class_label(df["target_amount_tableau"]).astype("category")

# 学習に使わない列を除外
drop_cols = [
    "KEY", "target_amount_tableau", "RST_TITLE", "menu",  # ID/目的変数/高カーディナリティ文章は除外
    # 必要に応じて "HOME_PAGE_URL" や "PHONE_NUM" を残す/落とす
]
use_cols = [c for c in df.columns if c not in drop_cols + ["y_class"]]

# 型調整（object→category、bool系はそのまま/0-1化）
X = df[use_cols].copy()

# object列は LightGBM に渡すとき category に変換しておくと楽です
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

y = df["y_class"]

# =========================================
# 2) OOF（Out-of-Fold）でクラス確率を推定
# =========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_proba = np.zeros((len(X), 3), dtype=float)  # [p_low, p_middle, p_high]
models = []

# 不均衡への対処：クラス重み（頻度の逆数）を自動計算
class_counts = y.value_counts()
class_weight = {cls: (len(y) / (len(class_counts) * cnt)) for cls, cnt in class_counts.items()}

for fold, (tr, va) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    clf = LGBMClassifier(
        objective="multiclass",
        learning_rate=0.05,
        n_estimators=3000,
        num_leaves=127,
        min_data_in_leaf=10,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        random_state=42 + fold,
        class_weight=class_weight,   # 重み付けで high を取り逃がしにくく
    )
    # LightGBM は pandas の category dtype をそのまま扱えます
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="multi_logloss", verbose=False)
    models.append(clf)

    proba = clf.predict_proba(X_va)  # 列順は clf.classes_ の並び
    # 列順を [low, middle, high] に並び替え
    idx_map = [list(clf.classes_).index(k) for k in ["low", "middle", "high"]]
    oof_proba[va, :] = proba[:, idx_map]

# OOF のざっくり評価（high を重視するなら macro F1 など）
y_pred_oof = np.array(["low", "middle", "high"])[oof_proba.argmax(axis=1)]
print(classification_report(y, y_pred_oof, digits=3))
print("Macro F1:", f1_score(y, y_pred_oof, average="macro"))

# 学習完了後の「全学習データでの最終モデル」（推論用）
final_clf = LGBMClassifier(
    objective="multiclass",
    learning_rate=0.05,
    n_estimators=int(np.mean([m.best_iteration_ or 3000 for m in models])),  # OOFでの平均反映（任意）
    num_leaves=127, min_data_in_leaf=10, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=1, random_state=777, class_weight=class_weight
)
final_clf.fit(X, y, verbose=False)
