import pandas as pd
import re
import os

# 1. 데이터 로드
df_raw = pd.read_csv("data/gta_balanced_100k.csv") # 이미 필터링된 파일 사용
gta4_sample = df_raw[df_raw['game'] == 'GTA4'].sample(n=1500, random_state=42)
gta5_sample = df_raw[df_raw['game'] == 'GTA5'].sample(n=1500, random_state=42)
df = pd.concat([gta4_sample, gta5_sample]).reset_index(drop=True)


# 2. 기계 자동 라벨링 규칙 (관련없음 추가)
def auto_label(row):
    text = str(row['review']).lower()

    # [추가] 관련없음(3) 분류 로직
    # 예시: 너무 짧거나, 단순 반복, 혹은 게임과 무관해 보이는 단어 포함 시
    if len(text) < 10 or any(word in text for word in ['link', 'http', 'discord', 'steamgroup']):
        return 3

    # 중립(1) 분류
    if any(kw in text for kw in ['average', 'mediocre', 'ok', 'decent', 'but', 'however']):
        return 1

    # 긍정(2) / 부정(0) 분류
    return 2 if row['voted_up'] == True else 0


df['label'] = df.apply(auto_label, axis=1)
if not os.path.exists('data'): os.makedirs('data')
df[['game', 'review', 'label']].to_csv("data/auto_labeled_result.csv", index=False, encoding='utf-8-sig')
print("✅ 기계 라벨링 완료: data/auto_labeled_result.csv")