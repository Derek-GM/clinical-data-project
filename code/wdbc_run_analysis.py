# /mnt/data/wdbc_run_analysis.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("/mnt/data/wdbc.csv")
OUT_DIR = Path("/mnt/data/wdbc_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst",
]

COLS = ["id", "diagnosis"] + FEATURES


def format_mean_sd(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH, header=None, names=COLS)
    df["y"] = (df["diagnosis"] == "M").astype(int)

    # 2) Missing rate
    missing_rate = df.isna().mean().rename("missing_rate").to_frame()
    missing_rate.to_csv(OUT_DIR / "missing_rate.csv", index=True)

    # 3) Table 1 baseline + t-test p-values (use selected key vars for readability)
    key_vars = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "concavity_mean", "concave_points_mean",
        "radius_worst", "perimeter_worst", "area_worst",
    ]
    rows = []
    df_b = df[df["y"] == 0]
    df_m = df[df["y"] == 1]
    for v in key_vars:
        xb = df_b[v].dropna()
        xm = df_m[v].dropna()
        t_stat, p_val = stats.ttest_ind(xm, xb, equal_var=False)  # Welch
        rows.append({
            "Variable": v,
            "Benign_mean_sd": format_mean_sd(xb),
            "Malignant_mean_sd": format_mean_sd(xm),
            "p_value": p_val,
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(OUT_DIR / "table1_stats.csv", index=False)

    # Write LaTeX for Table 1
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Baseline characteristics by diagnosis (Welch's t-test)}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Variable & Benign (B) & Malignant (M) & p-value \\\\")
    lines.append("\\midrule")
    for _, r in table1.iterrows():
        var_tex = str(r["Variable"]).replace("_", "\\_")
        lines.append(f"{var_tex} & {r['Benign_mean_sd']} & {r['Malignant_mean_sd']} & {format_p(float(r['p_value']))} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (OUT_DIR / "table1_baseline.tex").write_text("\n".join(lines), encoding="utf-8")

    # 4) Split
    X = df[FEATURES].values
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 5) Models
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42))
    ])
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
    }

    metrics_rows = []
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)

        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        metrics_rows.append({
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)

        # save confusion matrix plot for RF (use one figure)
        if name == "RandomForest":
            fig = plt.figure()
            plt.imshow(cm)
            plt.title("Confusion Matrix (Random Forest)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "fig_cm_rf.png", dpi=200)
            plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    # 6) ROC plot
    fig = plt.figure()
    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_roc.png", dpi=200)
    plt.close(fig)

    # 7) RF feature importance
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    top_features = [FEATURES[i] for i in idx]
    top_importances = importances[idx]

    fig = plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_features))[::-1], top_importances[::-1])
    plt.yticks(range(len(top_features))[::-1], [f.replace("_", " ") for f in top_features[::-1]])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances (Random Forest)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_rf_importance.png", dpi=200)
    plt.close(fig)

    # 8) Write LaTeX snippet for metrics + figures
    m_lr = metrics_df[metrics_df["Model"] == "LogisticRegression"].iloc[0]
    m_rf = metrics_df[metrics_df["Model"] == "RandomForest"].iloc[0]

    latex = []
    latex.append("\\subsection{Predictive Modeling Results}")
    latex.append(f"Logistic Regression achieved an accuracy of {m_lr['Accuracy']:.3f} and ROC-AUC of {m_lr['ROC_AUC']:.3f}. ")
    latex.append(f"Random Forest achieved an accuracy of {m_rf['Accuracy']:.3f} and ROC-AUC of {m_rf['ROC_AUC']:.3f}. ")
    latex.append("")
    latex.append("\\begin{figure}[H]")
    latex.append("\\centering")
    latex.append("\\includegraphics[width=0.78\\textwidth]{figures/fig_roc.png}")
    latex.append("\\caption{ROC curves for Logistic Regression and Random Forest on the test set.}")
    latex.append("\\end{figure}")
    latex.append("")
    latex.append("\\begin{figure}[H]")
    latex.append("\\centering")
    latex.append("\\includegraphics[width=0.62\\textwidth]{figures/fig_cm_rf.png}")
    latex.append("\\caption{Confusion matrix of Random Forest classifier on the test set.}")
    latex.append("\\end{figure}")
    latex.append("")
    latex.append("\\begin{figure}[H]")
    latex.append("\\centering")
    latex.append("\\includegraphics[width=0.85\\textwidth]{figures/fig_rf_importance.png}")
    latex.append("\\caption{Top 15 feature importances from the Random Forest model.}")
    latex.append("\\end{figure}")

    (OUT_DIR / "latex_metrics_and_figures.tex").write_text("\n".join(latex), encoding="utf-8")

    print("DONE")
    print(f"Samples={df.shape[0]}, features={len(FEATURES)}, M={(df['y']==1).sum()}, B={(df['y']==0).sum()}")
    print(metrics_df)


if __name__ == "__main__":
    main()

