"""Print OpenRouter key usage and account balance."""
import os, requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
h = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
k = requests.get("https://openrouter.ai/api/v1/auth/key", headers=h, timeout=30).json()["data"]
c = requests.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=30).json()["data"]
print(f"key usage ${k['usage']:.2f} | balance ${c['total_credits']-c['total_usage']:.2f}")
