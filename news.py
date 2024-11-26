import requests
from datetime import datetime, timedelta
import pandas as pd
import os

output_path = "data/crypto_compare.csv"
if os.path.exists(output_path):
    os.remove(output_path)

url = "https://data-api.cryptocompare.com/news/v1/article/list"

start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 10, 31, 16)

current_time = end_date
while current_time >= start_date:
    print(current_time)

    params = {
        "categories": "BTC",
        "sortOrder": "latest",
        "to_ts": int(current_time.timestamp()),
    }

    response = requests.get(url, params=params)
    data = response.json()["Data"]

    needed_data = []
    for news in data:
        needed_data.append(
            {
                "date": datetime.fromtimestamp(news["PUBLISHED_ON"]),
                "title": news["TITLE"],
                "content": news["BODY"],
                "sentiment": news["SENTIMENT"].lower(),
            }
        )

    current_time = needed_data[-1]["date"] - timedelta(seconds=1)

    needed_data = pd.DataFrame(needed_data)

    if not os.path.exists(output_path):
        needed_data.to_csv(output_path, mode="w", index=False)
    else:
        needed_data.to_csv(output_path, mode="a", index=False, header=False)
