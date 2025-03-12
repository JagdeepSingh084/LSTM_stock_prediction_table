# post.py
import requests

url = "http://127.0.0.1:5000/predict"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Forecast Data:")
    for date, price in data["forecast"].items():
        print(f"{date}: {price}")
    print(f"Model Accuracy: {data['accuracy_percentage']}%")
else:
    print(f"Error: {response.status_code} - {response.text}")