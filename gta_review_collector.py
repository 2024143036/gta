import requests
import pandas as pd
import time

def fetch_large_balanced_data(app_id, game_name, target_count=50000):
    reviews_list = []
    cursor = '*'
    filtered_count = 0  # 삭제된 데이터 카운트

    print(f"\n🚀 [{game_name}] 데이터 수집 시작 (10자 미만은 자동 제외)...")

    while len(reviews_list) < target_count:
        params = {
            'json': 1, 'filter': 'all', 'language': 'english',
            'cursor': cursor, 'num_per_page': 100, 'sort': 'oldest'
        }

        try:
            res = requests.get(f"https://store.steampowered.com/appreviews/{app_id}", params=params, timeout=15).json()
        except Exception as e:
            time.sleep(5)
            continue

        reviews = res.get('reviews', [])
        if not reviews: break

        for rev in reviews:
            review_text = str(rev.get('review', '')).strip()

            # [수정] 10자 미만인 경우 리스트에 넣지 않고 건너뜀
            if len(review_text) < 10:
                filtered_count += 1
                continue

            if len(reviews_list) < target_count:
                reviews_list.append({
                    'game': game_name,
                    'review': review_text,
                    'voted_up': rev['voted_up'],
                    'timestamp': rev['timestamp_created']
                })

        print(f"👉 [{game_name}] 수집됨: {len(reviews_list)}건 | 필터링됨(10자 미만): {filtered_count}건")

        cursor = res.get('cursor')
        if not cursor or cursor == '*': break
        time.sleep(0.5)

    return pd.DataFrame(reviews_list)


# 실행
df_gta4 = fetch_large_balanced_data("12210", "GTA4", target_count=50000)
df_gta5 = fetch_large_balanced_data("271590", "GTA5", target_count=50000)

final_df = pd.concat([df_gta4, df_gta5])
final_df.to_csv("gta_balanced_100k.csv", index=False, encoding='utf-8-sig')
print(f"\n✨ 완료! 총 {len(final_df)}건의 정제된 데이터셋 생성 완료.")