import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GTA 리뷰 분석 그래프 PNG 저장 코드
# ----------------------------------------------------------
# 이 코드 파일이 들어있는 "같은 폴더" 안에 charts 폴더를 만들고
# 그 안에 그래프 이미지를 저장합니다.
#
# 예:
# 현재 파일 위치:
# C:/Users/Administrator/Downloads/GTA2/GTA/GTA/make_gta_images_same_folder.py
#
# 저장 위치:
# C:/Users/Administrator/Downloads/GTA2/GTA/GTA/charts/
# ==========================================================


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"
CHART_DIR = BASE_DIR / "charts"

LABEL_NAMES = {
    0: "Negative(0)",
    1: "Neutral(1)",
    2: "Positive(2)"
}


def ensure_dirs():
    CHART_DIR.mkdir(exist_ok=True)


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def preprocess_label_df(df):
    if df is None:
        return None

    if "review" not in df.columns and "clean_review" in df.columns:
        df["review"] = df["clean_review"]

    if "game" not in df.columns:
        df["game"] = "Unknown"

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1, 2])]
    df["label"] = df["label"].astype(int)

    return df


def save_auto_label_distribution(auto_df):
    counts = auto_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)

    plt.figure(figsize=(7, 5))
    plt.bar([LABEL_NAMES[i] for i in [0, 1, 2]], counts.values)
    plt.title("Auto Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "01_auto_label_distribution.png", dpi=300)
    plt.close()


def save_manual_label_distribution(manual_df):
    counts = manual_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)

    plt.figure(figsize=(7, 5))
    plt.bar([LABEL_NAMES[i] for i in [0, 1, 2]], counts.values)
    plt.title("Manual Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "02_manual_label_distribution.png", dpi=300)
    plt.close()


def save_auto_manual_label_distribution(auto_df, manual_df):
    auto_counts = auto_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    manual_counts = manual_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)

    x = np.arange(3)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, auto_counts.values, width, label="Auto Label")
    plt.bar(x + width / 2, manual_counts.values, width, label="Manual Label")
    plt.title("Auto vs Manual Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(x, [LABEL_NAMES[i] for i in [0, 1, 2]])
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "03_auto_manual_label_distribution.png", dpi=300)
    plt.close()


def save_auto_label_by_game(auto_df):
    pivot = pd.crosstab(auto_df["game"], auto_df["label"]).reindex(columns=[0, 1, 2], fill_value=0)

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(9, 5))
    for idx, label_code in enumerate([0, 1, 2]):
        plt.bar(x + (idx - 1) * width, pivot[label_code].values, width, label=LABEL_NAMES[label_code])

    plt.title("Auto Label Distribution by Game")
    plt.xlabel("Game")
    plt.ylabel("Count")
    plt.xticks(x, pivot.index)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "04_auto_label_by_game.png", dpi=300)
    plt.close()


def save_manual_label_by_game(manual_df):
    pivot = pd.crosstab(manual_df["game"], manual_df["label"]).reindex(columns=[0, 1, 2], fill_value=0)

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(9, 5))
    for idx, label_code in enumerate([0, 1, 2]):
        plt.bar(x + (idx - 1) * width, pivot[label_code].values, width, label=LABEL_NAMES[label_code])

    plt.title("Manual Label Distribution by Game")
    plt.xlabel("Game")
    plt.ylabel("Count")
    plt.xticks(x, pivot.index)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "05_manual_label_by_game.png", dpi=300)
    plt.close()


def save_confusion_and_agreement(auto_df, manual_df):
    n = min(len(auto_df), len(manual_df))
    a = auto_df.iloc[:n].reset_index(drop=True)
    m = manual_df.iloc[:n].reset_index(drop=True)

    compare_df = pd.DataFrame({
        "auto_label": a["label"],
        "manual_label": m["label"]
    })

    confusion = pd.crosstab(
        compare_df["manual_label"],
        compare_df["auto_label"],
        rownames=["Manual Label"],
        colnames=["Auto Label"]
    ).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion.values)
    plt.title("Auto vs Manual Label Confusion Matrix")
    plt.xlabel("Auto Label")
    plt.ylabel("Manual Label")
    plt.xticks(np.arange(3), [LABEL_NAMES[i] for i in [0, 1, 2]], rotation=20, ha="right")
    plt.yticks(np.arange(3), [LABEL_NAMES[i] for i in [0, 1, 2]])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(confusion.values[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "06_auto_manual_confusion_matrix.png", dpi=300)
    plt.close()

    same_count = int((compare_df["auto_label"] == compare_df["manual_label"]).sum())
    diff_count = int(n - same_count)
    agreement_rate = same_count / n if n > 0 else 0

    plt.figure(figsize=(6, 5))
    plt.bar(["Same", "Different"], [same_count, diff_count])
    plt.title(f"Auto vs Manual Agreement ({agreement_rate:.4f})")
    plt.xlabel("Comparison")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "07_auto_manual_agreement.png", dpi=300)
    plt.close()


def save_auto_train_loss(auto_history):
    plt.figure(figsize=(8, 5))
    plt.plot(auto_history["epoch"], auto_history["train_loss"], marker="o")
    plt.title("Auto Label Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xticks(auto_history["epoch"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "08_auto_train_loss.png", dpi=300)
    plt.close()


def save_manual_train_loss(manual_history):
    plt.figure(figsize=(8, 5))
    plt.plot(manual_history["epoch"], manual_history["train_loss"], marker="o")
    plt.title("Manual Label Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xticks(manual_history["epoch"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "09_manual_train_loss.png", dpi=300)
    plt.close()


def save_valid_accuracy_compare(auto_history, manual_history):
    plt.figure(figsize=(8, 5))
    plt.plot(auto_history["epoch"], auto_history["valid_accuracy"], marker="o", label="Auto Label")
    plt.plot(manual_history["epoch"], manual_history["valid_accuracy"], marker="o", label="Manual Label")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.xticks(auto_history["epoch"])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "10_valid_accuracy_comparison.png", dpi=300)
    plt.close()


def save_train_accuracy_compare(auto_history, manual_history):
    plt.figure(figsize=(8, 5))
    plt.plot(auto_history["epoch"], auto_history["train_accuracy"], marker="o", label="Auto Label")
    plt.plot(manual_history["epoch"], manual_history["train_accuracy"], marker="o", label="Manual Label")
    plt.title("Train Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.xticks(auto_history["epoch"])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "11_train_accuracy_comparison.png", dpi=300)
    plt.close()


def save_final_valid_accuracy_bar(auto_history, manual_history):
    final_auto = float(auto_history.iloc[-1]["valid_accuracy"])
    final_manual = float(manual_history.iloc[-1]["valid_accuracy"])

    plt.figure(figsize=(7, 5))
    plt.bar(["Auto Label", "Manual Label"], [final_auto, final_manual])
    plt.title("Final Validation Accuracy")
    plt.xlabel("Training Data")
    plt.ylabel("Final Validation Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "12_final_valid_accuracy_bar.png", dpi=300)
    plt.close()


def main():
    ensure_dirs()

    auto_df = load_csv(DATA_DIR / "auto_labeled_result.csv")
    manual_df = load_csv(DATA_DIR / "manual_labeling_target.csv")
    auto_history = load_csv(RESULT_DIR / "mobilebert_auto_history.csv")
    manual_history = load_csv(RESULT_DIR / "mobilebert_manual_history.csv")

    if auto_df is None:
        print("❌ data/auto_labeled_result.csv 파일이 없습니다.")
        return

    if manual_df is None:
        print("❌ data/manual_labeling_target.csv 파일이 없습니다.")
        return

    if auto_history is None:
        print("❌ result/mobilebert_auto_history.csv 파일이 없습니다.")
        return

    if manual_history is None:
        print("❌ result/mobilebert_manual_history.csv 파일이 없습니다.")
        return

    auto_df = preprocess_label_df(auto_df)
    manual_df = preprocess_label_df(manual_df)

    save_auto_label_distribution(auto_df)
    save_manual_label_distribution(manual_df)
    save_auto_manual_label_distribution(auto_df, manual_df)
    save_auto_label_by_game(auto_df)
    save_manual_label_by_game(manual_df)
    save_confusion_and_agreement(auto_df, manual_df)
    save_auto_train_loss(auto_history)
    save_manual_train_loss(manual_history)
    save_valid_accuracy_compare(auto_history, manual_history)
    save_train_accuracy_compare(auto_history, manual_history)
    save_final_valid_accuracy_bar(auto_history, manual_history)

    print("✅ 그래프 이미지 저장 완료")
    print("저장 폴더:", CHART_DIR)
    print("\n생성된 파일:")
    print("01_auto_label_distribution.png")
    print("02_manual_label_distribution.png")
    print("03_auto_manual_label_distribution.png")
    print("04_auto_label_by_game.png")
    print("05_manual_label_by_game.png")
    print("06_auto_manual_confusion_matrix.png")
    print("07_auto_manual_agreement.png")
    print("08_auto_train_loss.png")
    print("09_manual_train_loss.png")
    print("10_valid_accuracy_comparison.png")
    print("11_train_accuracy_comparison.png")
    print("12_final_valid_accuracy_bar.png")


if __name__ == "__main__":
    main()
