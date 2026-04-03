import os, json
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONA_DIR = os.path.join(BASE_DIR, "data", "personas")

@lru_cache(maxsize=64)
def list_persona_keys():
    if not os.path.isdir(PERSONA_DIR):
        return []
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(PERSONA_DIR)
        if f.endswith(".json")
    ])

@lru_cache(maxsize=128)
def load_persona(key: str) -> dict | None:
    path = os.path.join(PERSONA_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)