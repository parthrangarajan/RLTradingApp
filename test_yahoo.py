import requests

url = "https://query1.finance.yahoo.com/v8/finance/chart/GME"
params = {"period1": 1572566400, "period2": 1615593600, "interval": "1d"}
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}

res = requests.get(url, params=params, headers=headers, timeout=30)
print("HTTP status:", res.status_code)
data = res.json()
result = (data.get("chart", {}).get("result") or [None])[0]
print("Trading days returned:", len(result["timestamp"]) if result and result.get("timestamp") else 0)