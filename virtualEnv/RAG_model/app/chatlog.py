# chatlog.py
import os, json, time
from typing import List, Dict

BASE = "virtualEnv/RAG_model/app/data/chatlogs"
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
