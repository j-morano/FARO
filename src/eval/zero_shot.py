import json
import re
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, classification_report
)


def norm_txt(t):
    return t.lower().replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").strip()


# Priority-ordered: check most disease-specific findings first, since
# generic terms like "trocken" can co-occur with a specific pathology
# (e.g. "Drusen, trocken" should be classified as DRU, not NORMAL).
CLASS_KEYWORDS = [
    ("GA", ["geographische atrophie", "atrophie", "atrophisch", "pigmentepithel-atrophie"]),
    ("ERM", ["epiretinale membran", "erm", "membran"]),
    ("DRU", ["drusen", "drusenoide"]),
    ("DME", ["oedem", "makulaoedem", "zysten", "zyste", "zystoides", "cystoides", "cme", "srf"]),
    ("CNV", ["subretinale fluessigkeit", "pigmentepithelabhebung", "ped", "neovask"]),
    ("NORMAL", ["trocken", "unauffaellig", "ohne befund", "normalzustand", "regularitaet normal"]),
]


def classify_prediction(text):
    norm = norm_txt(text)
    for class_name, keywords in CLASS_KEYWORDS:
        for kw in keywords:
            kw_norm = norm_txt(kw)
            if re.search(rf"\b{re.escape(kw_norm)}", norm):
                return class_name
    return "UNKNOWN"  # no keyword matched at all


def evaluate_generative_classification(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)  # no split filtering — use everything

    df["predicted_class"] = df["prediction"].apply(classify_prediction)

    n_unknown = (df["predicted_class"] == "UNKNOWN").sum()
    print(f"N={len(df)} | Unmatched (UNKNOWN): {n_unknown} "
          f"({100*n_unknown/len(df):.1f}%)")

    # For metrics, keep UNKNOWN as a distinct wrong prediction rather
    # than dropping, so unmatched generations are penalized (not
    # silently excluded)
    y_true = df["class_name"].values
    y_pred = df["predicted_class"].values

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nAccuracy: {acc:.4f} | Balanced Accuracy: {bacc:.4f} | MCC: {mcc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm_df)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "n_unknown": int(n_unknown),
        "confusion_matrix": cm_df,
        "df": df,
    }


if __name__ == "__main__":
    results = evaluate_generative_classification(
        "/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/__posttraining/infer/nv05/OCTAVE__r-axbidmlstm_lp-bidmlstm_nq-16_as-32_sk-True_fw-True__010.json",
    )
