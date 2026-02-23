import requests
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

class NewsAdjuster:
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY", "")
        self.base_url = "https://finnhub.io/api/v1/news?category=forex"

    def get_market_sentiment(self):
        """Fetch latest forex news and calculate global sentiment score."""
        if not self.api_key:
            return 0.5 # Neutral if no API key
            
        try:
            response = requests.get(f"{self.base_url}&token={self.api_key}")
            news = response.json()
            
            if not news:
                return 0.5
                
            sentiments = []
            for item in news[:20]: # Look at last 20 headlines
                analysis = TextBlob(item.get('headline', ''))
                sentiments.append(analysis.sentiment.polarity)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            # Map sentiment (-1 to 1) to risk factor (0.5 to 1.5)
            # 1.0 is normal risk. Negative sentiment reduces risk.
            risk_multiplier = 1.0 + (avg_sentiment * 0.5)
            return round(max(0.5, min(risk_multiplier, 1.5)), 2)
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return 1.0 # Default to neutral risk on error

    def get_symbol_specific_sentiment(self, symbol):
        """Fetch sentiment for a specific symbol (e.g., BTC, EUR)."""
        # Finnhub general news doesn't always have symbol-specific markers in the free tier
        # but we can filter headlines.
        # Implementation omitted for brevity, returns global for now.
        return self.get_market_sentiment()
