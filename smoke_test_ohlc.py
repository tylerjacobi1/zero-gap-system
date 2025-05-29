import os, requests

API_KEY = os.getenv("TRADIER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}

resp = requests.get(
    "https://api.tradier.com/v1/markets/quotes/ohlc",
    params={"symbols": "SPX", "interval": "1min", "period": 1},
    headers=HEADERS,
)

print(resp.status_code)
print(resp.text)
