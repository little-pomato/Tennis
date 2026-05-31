import requests
import sys
import os

def test_health():
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url)
        print(f"Health Check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Health Check Failed: {e}")

def test_upload(video_path):
    url = "http://localhost:8000/upload"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    print(f"Uploading {video_path}...")
    files = {"file": open(video_path, "rb")}
    try:
        response = requests.post(url, files=files)
        print(f"Upload Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print("Successfully received results!")
            print(f"Metadata: {results.get('metadata')}")
            # Save to a local file for inspection
            with open("api_test_results.json", "w") as f:
                import json
                json.dump(results, f, indent=4)
            print("Results saved to api_test_results.json")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Upload Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_upload(sys.argv[1])
    else:
        print("Usage: python test_api.py <path_to_video>")
        test_health()
