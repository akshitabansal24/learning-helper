import requests
from win32com.client import Dispatch

# The API endpoint
scanUrl = "https://learning-helper-2025-451212.uc.r.appspot.com/scan"
genAIUrl = "https://learning-helper-2025-451212.uc.r.appspot.com/genAI"
speak = Dispatch("SAPI.SpVoice")

response = requests.get(scanUrl)
response_json = response.json()
text = (response_json['data'])
print(text)

response = requests.get(genAIUrl)
response_json = response.json()
speak.Speak(response_json['data'])
