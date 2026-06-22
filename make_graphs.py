import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# GTA4/GTA5 리뷰 분석 그래프 생성 코드
# ----------------------------------------------------------
# 필수 파일:
# data/auto_labeled_result.csv
# result/mobilebert_auto_history.csv
#
# 있으면 자동으로 같이 사용하는 파일:
# data/manual_labeling_target.csv
# result/mobilebert_manual_history.csv
#
# 생성 위치:
# result/charts/
# ==========================================================


LABEL_NAMES = {
    0: "Negative(0)",
    1: "Neutral(1)",
    2: "Positive(2)"
}


def make_dirs():
    os.makedirs("result", exist_ok=True)
    os.makedirs("result/charts", exist_ok=True)


def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def clean_label_df(df):
    if df is None:
        return None

    # review 컬럼 이름 보정
    if "review" not in df.columns and "clean_review" in df.columns:
        df["review"] = df["clean_review"]

    # game 컬럼이 없으면 Unknown 처리
    if "game" not in df.columns:
        df["game"] = "Unknown"

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1, 2])]
    df["label"] = df["label"].astype(int)

    return df


def chart_auto_label_distribution(auto_df):
    counts = auto_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)

    plt.figure(figsize=(7, 5))
    plt.bar([LABEL_NAMES[i] for i in [0, 1, 2]], counts.values)
    plt.title("Auto Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("result/charts/01_auto_label_distribution.png", dpi=300)
    plt.close()

    out_df = pd.DataFrame({
        "label": [LABEL_NAMES[i] for i in [0, 1, 2]],
        "count": counts.values
    })
    out_df.to_csv("result/auto_label_distribution.csv", index=False, encoding="utf-8-sig")


def chart_auto_label_by_game(auto_df):
    pivot = pd.crosstab(auto_df["game"], auto_df["label"]).reindex(columns=[0, 1, 2], fill_value=0)

    x = np.arange(len(pivot.index))
    width = 0.25

    plt.figure(figsize=(9, 5))
    for idx, label_code in enumerate([0, 1, 2]):
        plt.bar(
            x + (idx - 1) * width,
            pivot[label_code].values,
            width,
            label=LABEL_NAMES[label_code]
        )

    plt.title("Auto Label Distribution by Game")
    plt.xlabel("Game")
    plt.ylabel("Count")
    plt.xticks(x, pivot.index)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/charts/02_auto_label_by_game.png", dpi=300)
    plt.close()

    pivot.to_csv("result/auto_label_by_game.csv", encoding="utf-8-sig")


def chart_train_loss(auto_history):
    plt.figure(figsize=(8, 5))
    plt.plot(auto_history["epoch"], auto_history["train_loss"], marker="o", label="Auto Label")
    plt.title("MobileBERT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xticks(auto_history["epoch"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/charts/03_train_loss_auto.png", dpi=300)
    plt.close()


def chart_train_valid_accuracy(auto_history):
    plt.figure(figsize=(8, 5))
    plt.plot(auto_history["epoch"], auto_history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(auto_history["epoch"], auto_history["valid_accuracy"], marker="o", label="Validation Accuracy")
    plt.title("MobileBERT Train / Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(auto_history["epoch"])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/charts/04_train_valid_accuracy_auto.png", dpi=300)
    plt.close()


def chart_auto_manual_label_distribution(auto_df, manual_df):
    if manual_df is None:
        print("⚠️ 수동 라벨 파일이 없어서 자동/수동 라벨 분포 비교 그래프는 건너뜁니다.")
        return

    auto_counts = auto_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    manual_counts = manual_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)

    x = np.arange(3)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, auto_counts.values, width, label="Auto Label")
    plt.bar(x + width / 2, manual_counts.values, width, label="Manual Label")
    plt.title("Auto Label vs Manual Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(x, [LABEL_NAMES[i] for i in [0, 1, 2]])
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/charts/05_auto_manual_label_distribution.png", dpi=300)
    plt.close()

    out_df = pd.DataFrame({
        "label": [LABEL_NAMES[i] for i in [0, 1, 2]],
        "auto_count": auto_counts.values,
        "manual_count": manual_counts.values
    })
    out_df.to_csv("result/auto_manual_label_distribution.csv", index=False, encoding="utf-8-sig")


def chart_auto_manual_accuracy_compare(auto_history, manual_history):
    if manual_history is None:
        print("⚠️ 수동 학습 결과 파일이 없어서 자동/수동 정확도 비교 그래프는 건너뜁니다.")
        return

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
    plt.savefig("result/charts/06_valid_accuracy_auto_vs_manual.png", dpi=300)
    plt.close()

    final_auto = float(auto_history.iloc[-1]["valid_accuracy"])
    final_manual = float(manual_history.iloc[-1]["valid_accuracy"])

    plt.figure(figsize=(7, 5))
    plt.bar(["Auto Label", "Manual Label"], [final_auto, final_manual])
    plt.title("Final Validation Accuracy")
    plt.xlabel("Training Data")
    plt.ylabel("Final Validation Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("result/charts/07_final_valid_accuracy_auto_vs_manual.png", dpi=300)
    plt.close()

    out_df = pd.DataFrame({
        "dataset": ["auto_label", "manual_label"],
        "final_valid_accuracy": [final_auto, final_manual]
    })
    out_df.to_csv("result/final_valid_accuracy_compare.csv", index=False, encoding="utf-8-sig")


def chart_auto_manual_confusion(auto_df, manual_df):
    if manual_df is None:
        print("⚠️ 수동 라벨 파일이 없어서 혼동행렬 그래프는 건너뜁니다.")
        return

    n = min(len(auto_df), len(manual_df))
    a = auto_df.iloc[:n].reset_index(drop=True)
    m = manual_df.iloc[:n].reset_index(drop=True)

    compare_df = pd.DataFrame({
        "game": a["game"],
        "review": a["review"] if "review" in a.columns else "",
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
    plt.xticks(np.arange(3), [LABEL_NAMES[i] for i in [0, 1, 2]], rotation=25, ha="right")
    plt.yticks(np.arange(3), [LABEL_NAMES[i] for i in [0, 1, 2]])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(confusion.values[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig("result/charts/08_auto_manual_confusion_matrix.png", dpi=300)
    plt.close()

    compare_df["same_label"] = compare_df["auto_label"] == compare_df["manual_label"]
    same_count = int(compare_df["same_label"].sum())
    diff_count = int(n - same_count)
    agreement_rate = same_count / n if n > 0 else 0

    plt.figure(figsize=(6, 5))
    plt.bar(["Same", "Different"], [same_count, diff_count])
    plt.title(f"Auto vs Manual Label Agreement: {agreement_rate:.4f}")
    plt.xlabel("Comparison")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("result/charts/09_auto_manual_agreement.png", dpi=300)
    plt.close()

    compare_df.to_csv("result/auto_manual_label_compare.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv("result/auto_manual_confusion_matrix.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame({
        "total_count": [n],
        "same_count": [same_count],
        "different_count": [diff_count],
        "agreement_rate": [agreement_rate]
    })
    summary_df.to_csv("result/auto_manual_agreement_summary.csv", index=False, encoding="utf-8-sig")


def main():
    make_dirs()

    auto_label_path = "data/auto_labeled_result.csv"
    manual_label_path = "data/manual_labeling_target.csv"

    auto_history_path = "result/mobilebert_auto_history.csv"
    manual_history_path = "result/mobilebert_manual_history.csv"

    auto_df = load_csv_if_exists(auto_label_path)
    auto_history = load_csv_if_exists(auto_history_path)

    if auto_df is None:
        print(f"❌ 자동 라벨 파일이 없습니다: {auto_label_path}")
        return

    if auto_history is None:
        print(f"❌ 자동 학습 결과 파일이 없습니다: {auto_history_path}")
        return

    manual_df = load_csv_if_exists(manual_label_path)
    manual_history = load_csv_if_exists(manual_history_path)

    auto_df = clean_label_df(auto_df)
    manual_df = clean_label_df(manual_df)

    print("✅ 자동 라벨 데이터:", len(auto_df), "건")
    print("✅ 자동 학습 결과:", auto_history_path)

    if manual_df is not None:
        print("✅ 수동 라벨 데이터:", len(manual_df), "건")
    else:
        print("⚠️ 수동 라벨 데이터는 아직 없습니다. 자동 라벨 그래프만 생성합니다.")

    if manual_history is not None:
        print("✅ 수동 학습 결과:", manual_history_path)
    else:
        print("⚠️ 수동 학습 결과는 아직 없습니다. 자동/수동 성능 비교 그래프는 나중에 생성됩니다.")

    # 자동 라벨만으로 만들 수 있는 그래프
    chart_auto_label_distribution(auto_df)
    chart_auto_label_by_game(auto_df)
    chart_train_loss(auto_history)
    chart_train_valid_accuracy(auto_history)

    # 수동 파일이 있으면 추가로 만드는 그래프
    chart_auto_manual_label_distribution(auto_df, manual_df)
    chart_auto_manual_accuracy_compare(auto_history, manual_history)
    chart_auto_manual_confusion(auto_df, manual_df)

    print("\n✅ 그래프 생성 완료")
    print("저장 위치: result/charts")
    print("\n[자동 라벨만 있어도 생성되는 그래프]")
    print("01_auto_label_distribution.png")
    print("02_auto_label_by_game.png")
    print("03_train_loss_auto.png")
    print("04_train_valid_accuracy_auto.png")

    print("\n[수동 결과까지 있으면 추가 생성되는 그래프]")
    print("05_auto_manual_label_distribution.png")
    print("06_valid_accuracy_auto_vs_manual.png")
    print("07_final_valid_accuracy_auto_vs_manual.png")
    print("08_auto_manual_confusion_matrix.png")
    print("09_auto_manual_agreement.png")


if __name__ == "__main__":
    main()
