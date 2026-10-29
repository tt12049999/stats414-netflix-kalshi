# STATS 414 Midterm: Netflix weekly #1 movie, Kalshi prediction market

Team: Yutong Ma, Ning Mu, Mike Wu

Question: do the past 24h price change and 24h volume improve winner prediction beyond the current price?

- `slides/`: midterm slides (pptx and pdf)
- `input_data/`: Kalshi hourly price data and weekly results (public API)
- `code/netflix_midterm_v2.py`: full analysis code
- `requirements.txt`: Python packages

Run:
```bash
pip install -r requirements.txt
cd code
python netflix_midterm_v2.py --data-dir ../input_data --output-dir ../results --figure-dir ../figures
```
