import pandas as pd

# 이미 필터링된 자동 라벨링 결과 파일 사용
df = pd.read_csv("data/auto_labeled_result.csv")

# 수동 라벨링을 위해 라벨을 -1로 초기화
df['label'] = -1

df[['game', 'review', 'label']].to_csv("data/manual_labeling_target.csv", index=False, encoding='utf-8-sig')
print("✅ 수동 라벨링 준비 완료: data/manual_labeling_target.csv")