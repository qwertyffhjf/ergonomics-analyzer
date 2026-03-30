#!/usr/bin/env python3
import cv2
import base64
import json
import requests
import sys
import os

if len(sys.argv) < 2:
    print("Usage: analyze_video.py <video_file>")
    sys.exit(1)

video_path = sys.argv[1]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not set")
    sys.exit(1)

MODEL = "gpt-4o-mini"

PROMPT = """
Ты — эксперт по эргономике. Анализируй фото рабочего места.
Ответь строго в формате JSON:
{
  "operation_type": "слесарная" | "сварочная" | "неопределено",
  "ergonomics": {
    "workstation_height": "good" | "bad",
    "tool_reach": "good" | "bad",
    "posture": "good" | "bad",
    "extra_movements": 0,
    "comment": "кратко"
  },
  "efficiency": {
    "posture_stable": true | false,
    "movement_optimized": true | false,
    "safety_observed": true | false
  }
}
"""

def analyze_frame(frame_bytes):
    base64_image = base64.b64encode(frame_bytes).decode('utf-8')
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return {"error": f"API error {response.status_code}", "details": response.text}
    try:
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        return {"error": "JSON parse error", "raw": content, "exception": str(e)}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps))
    frame_count = 0
    prev_frame = None
    results = []

    print(f"Обработка видео: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval != 0:
            frame_count += 1
            continue
        frame_count += 1

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            non_zero = cv2.countNonZero(gray)
            if non_zero < 5000:
                continue
        prev_frame = frame.copy()

        _, img_bytes = cv2.imencode('.jpg', frame)
        frame_data = img_bytes.tobytes()
        print(f"Анализ кадра {frame_count}...")
        result = analyze_frame(frame_data)
        result['frame_number'] = frame_count
        results.append(result)

    cap.release()
    return results

if __name__ == "__main__":
    if not os.path.exists(video_path):
        print(f"Файл не найден: {video_path}")
        sys.exit(1)

    results = process_video(video_path)

    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Готово. Отчёт сохранён в report.json")
