import os
import requests

API_KEY = os.getenv("TRADIER_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}

symbol     = "SPX"
expiration = "2025-06-20"  # pick a real date from your expirations list

resp = requests.get(
    "https://api.tradier.com/v1/markets/options/chains",
    params={
        "symbol":     symbol,
        "expiration": expiration,
        "greeks":     "true",
    },
    headers=HEADERS,
)
resp.raise_for_status()
opt0 = resp.json()["options"]["option"][0]
print("Available keys on first option:", opt0.keys())
