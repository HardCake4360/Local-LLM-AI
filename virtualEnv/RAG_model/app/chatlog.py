# chatlog.py
import os, json, time
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(BASE_DIR, "data", "chatlogs")
os.makedirs(BASE, exist_ok=True)

def _path(user_id: str) -> str:
    return os.path.join(BASE, f"{user_id}.jsonl")

def append_log(user_id: str, speaker: str, text: str) -> None:
    rec = {"ts": time.time(), "speaker": speaker, "text": text}
    with open(_path(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_log(user_id: str, max_items: int = 200) -> List[Dict]:
    p = _path(user_id)
    if not os.path.exists(p):
        return []
    lines = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except:
                continue
    return lines[-max_items:]

def reset_log(user_id: str) -> None:
    p = _path(user_id)
    if os.path.exists(p):
        os.remove(p)
    reset_summary(user_id)

def _summary_path(user_id: str) -> str:
    return os.path.join(BASE, f"{user_id}.summary.json")

def write_summary(user_id: str, summary: dict) -> None:
    with open(_summary_path(user_id), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

def read_summary(user_id: str) -> dict | None:
    p = _summary_path(user_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def reset_summary(user_id: str) -> None:
    p = _summary_path(user_id)
    if os.path.exists(p):
        os.remove(p)