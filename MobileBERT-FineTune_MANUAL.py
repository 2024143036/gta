import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os


def main():
    # 0. GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치 : ", device)
    if device.type == "cpu":
        print("💡 안내: 현재 CPU로 구동 중입니다. 학습 속도가 다소 느릴 수 있습니다.")

    # 1. 학습 시 경고 메시지 제거
    logging.set_verbosity_error()

    # 2. 데이터 로드 및 요구사항에 맞춘 균등 샘플링
    path = "data/manual_labeling_target.csv"
    try:
        df = pd.read_csv(path)
        print(f"✅ 원본 데이터 로드 완료: 총 {len(df):,}건")
    except FileNotFoundError:
        # 파일명이 manual_labeling_target(3).csv인 경우 자동으로 다시 시도
        path = "data/manual_labeling_target(3).csv"
        try:
            df = pd.read_csv(path)
            print(f"✅ 원본 데이터 로드 완료: 총 {len(df):,}건")
        except FileNotFoundError:
            print("❌ 수동 라벨 파일이 존재하지 않습니다.")
            print("확인한 경로:")
            print(" - data/manual_labeling_target.csv")
            print(" - data/manual_labeling_target(3).csv")
            return

    # review 컬럼만 있고 clean_review가 없을 때 대비
    if "clean_review" not in df.columns and "review" in df.columns:
        df["clean_review"] = df["review"]

    if "clean_review" not in df.columns:
        print("❌ clean_review 또는 review 컬럼이 필요합니다.")
        print("현재 컬럼:", df.columns.tolist())
        return

    if "label" not in df.columns:
        print("❌ label 컬럼이 필요합니다.")
        print("현재 컬럼:", df.columns.tolist())
        return

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["clean_review", "label"])
    df = df[df["label"].isin([0, 1, 2])]
    df["label"] = df["label"].astype(int)

    print("\n=== 라벨별 원본 개수 ===")
    print(df["label"].value_counts().sort_index())

    target_train_size = 1000
    target_val_size = 200
    total_needed = target_train_size + target_val_size

    # 데이터가 부족한 라벨이 있으면 가능한 최소 개수에 맞춰 자동 조정
    label_counts = df["label"].value_counts()
    min_count = min(label_counts.get(0, 0), label_counts.get(1, 0), label_counts.get(2, 0))

    if min_count == 0:
        print("❌ 0, 1, 2 라벨 중 데이터가 0건인 라벨이 있습니다.")
        return

    if min_count < total_needed:
        print("\n⚠️ 라벨별 데이터가 1,200건보다 적은 라벨이 있어서 샘플 수를 자동 조정합니다.")
        print(f"가장 적은 라벨 개수: {min_count}건")

        target_val_size = max(1, int(min_count * 0.2))
        target_train_size = min_count - target_val_size
        total_needed = target_train_size + target_val_size

        print(f"조정된 학습 개수: 라벨별 {target_train_size}건")
        print(f"조정된 검증 개수: 라벨별 {target_val_size}건")

    train_df_list = []
    valid_df_list = []

    # 3개 클래스(0, 1, 2)에서 정확하게 동등한 개수의 실제 데이터 추출
    for label_code in [0, 1, 2]:
        pool = df[df['label'] == label_code]
        sampled = pool.sample(n=total_needed, random_state=2026)

        train_df_list.append(sampled.iloc[:target_train_size])
        valid_df_list.append(sampled.iloc[target_train_size:])

    # 병합 후 최종 무작위 셔플
    train_df = pd.concat(train_df_list).sample(frac=1, random_state=2026).reset_index(drop=True)
    valid_df = pd.concat(valid_df_list).sample(frac=1, random_state=2026).reset_index(drop=True)

    train_text = list(train_df["clean_review"].astype(str).values)
    train_labels = train_df["label"].values

    valid_text = list(valid_df["clean_review"].astype(str).values)
    valid_labels = valid_df["label"].values

    print("\n=== 데이터셋 확보 확인 ===")
    print(f"학습 데이터 개수: {len(train_df)}건 (라벨별 {target_train_size}건 균등)")
    print(f"검증 데이터 개수: {len(valid_df)}건 (라벨별 {target_val_size}건 균등)")

    # 3. 텍스트 데이터의 토큰화
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

    print("\n🔤 텍스트 데이터 토큰화 진행 중...")
    train_inputs = tokenizer(train_text, truncation=True, max_length=128, add_special_tokens=True, padding="max_length")
    valid_inputs = tokenizer(valid_text, truncation=True, max_length=128, add_special_tokens=True, padding="max_length")

    # 4. torch에 학습 시키기 위한 데이터셋 및 DataLoader 설정
    batch_size = 16

    train_ds = TensorDataset(torch.tensor(train_inputs['input_ids']),
                             torch.tensor(train_inputs['attention_mask']),
                             torch.tensor(train_labels))
    train_sampler = RandomSampler(train_ds)
    train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)

    valid_ds = TensorDataset(torch.tensor(valid_inputs['input_ids']),
                             torch.tensor(valid_inputs['attention_mask']),
                             torch.tensor(valid_labels))
    valid_sampler = SequentialSampler(valid_ds)
    valid_dataloader = DataLoader(valid_ds, sampler=valid_sampler, batch_size=batch_size)

    # 5. 사전학습 언어모델 설정
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    epoch = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=epoch * len(train_dataloader))

    # 6. 학습 및 검증 루프
    epoch_results = []

    for e in range(epoch):
        model.train()
        total_train_loss = 0.0
        process_bar = tqdm(train_dataloader, desc=f"Training Epoch {e + 1}", leave=False)

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch

            model.zero_grad()

            # Forward Pass
            outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            process_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- 학습 정확도 평가 ---
        model.eval()
        train_preds, train_true = [], []
        process_bar_t = tqdm(train_dataloader, desc=f"Evaluating Train Epoch {e + 1}", leave=False)
        for batch in process_bar_t:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        train_acc = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)

        # --- 검증 정확도 평가 ---
        valid_preds, valid_true = [], []
        progress_bar_v = tqdm(valid_dataloader, desc=f"Evaluating Valid Epoch {e + 1}", leave=False)
        for batch in progress_bar_v:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_mask, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())
        valid_acc = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)

        epoch_results.append((avg_train_loss, train_acc, valid_acc))
        print(f"Epoch {e + 1} 완료 -> 오차: {avg_train_loss:.4f} | 학습 정확도: {train_acc:.4f} | 검증 정확도: {valid_acc:.4f}")

    # 7. 최종 결과 출력
    print("\n=== 최종 학습 및 검증 결과 ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
        print(f"Epoch {idx} : 학습 오차 - {loss:.4f}, 학습 정확도 - {tacc:.4f}, 검증 정확도 - {vacc:.4f}")

    final_validation_accuracy = epoch_results[-1][2]
    print("\n==================================================")
    if final_validation_accuracy >= 0.85:
        print(f"🎉 목표 달성 성공! 최종 검증 정확도 {final_validation_accuracy:.4f}로 0.85를 넘겼습니다.")
    else:
        print(f"💡 현재 검증 정확도는 {final_validation_accuracy:.4f} 입니다.")
    print("==================================================")

    # 8. 모델 저장
    print("\n=== 모델 저장 ===")
    model.save_pretrained('best_gta_mobilebert_manual')
    tokenizer.save_pretrained('best_gta_mobilebert_manual')
    print(" 모델 가중치 및 토크나이저 저장 완료 ('best_gta_mobilebert_manual' 폴더)")

    # 9. 수동 라벨 학습 결과 저장
    print("\n=== 수동 라벨 학습 결과 CSV 저장 ===")
    os.makedirs("result", exist_ok=True)

    result_df = pd.DataFrame(
        epoch_results,
        columns=["train_loss", "train_accuracy", "valid_accuracy"]
    )
    result_df.insert(0, "epoch", range(1, len(result_df) + 1))
    result_df.insert(1, "dataset", "manual_label")

    result_df.to_csv(
        "result/mobilebert_manual_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = pd.DataFrame({
        "dataset": ["manual_label"],
        "final_train_loss": [epoch_results[-1][0]],
        "final_train_accuracy": [epoch_results[-1][1]],
        "final_valid_accuracy": [epoch_results[-1][2]],
        "train_count": [len(train_df)],
        "valid_count": [len(valid_df)],
        "train_per_label": [target_train_size],
        "valid_per_label": [target_val_size]
    })

    summary_df.to_csv(
        "result/mobilebert_manual_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("저장 완료:")
    print(" - result/mobilebert_manual_history.csv")
    print(" - result/mobilebert_manual_summary.csv")


if __name__ == "__main__":
    main()
