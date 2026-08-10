import os
import sys
import threading
import time
import webbrowser
import uvicorn

# Reconfigure stdout to UTF-8 to prevent Windows CP1252 encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("\n========================================================")
    print("  ISL TRANSLATOR — FASTAPI + UVICORN ENGINE")
    print("========================================================\n")

    threading.Thread(target=open_browser, daemon=True).start()

    print("  Real-Time Webcam: http://127.0.0.1:5000")
    print("  Dataset Tester:   http://127.0.0.1:5000/upload")
    print("  Collect Data:     http://127.0.0.1:5000/collect")
    print("  OpenAPI Specs:    http://127.0.0.1:5000/docs\n")

    uvicorn.run("src.backend:app", host="127.0.0.1", port=5000, reload=True)
