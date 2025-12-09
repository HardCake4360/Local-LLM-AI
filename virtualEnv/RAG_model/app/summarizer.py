from typing import List, Dict, Any
import re
from collections import Counter

def _extract_keyphrases(text: str, top_k: int = 8):
    toks = re.findall(r"[A-Za-z가-힣0-9]+", text)
    toks = [t for t in toks if len(t) > 1]
    return [w for w, _ in Counter(toks).most_common(top_k)]

def summarize_context(history: List[Dict[str, str]], recent_turns: int = 20, max_tokens: int = 384) -> Dict[str, Any]:
    recent = history[-recent_turns:] if recent_turns > 0 else history
    transcript = "\n".join(f"{m.get('speaker')}:{m.get('text','')}" for m in recent)

    # keyphrases + naive signals
    all_text = " ".join(m.get("text","") for m in recent)
    keyphrases = _extract_keyphrases(all_text, top_k=10)

    trust = 0.5
    low = all_text.lower()
    if any(k in low for k in ["고마워","thanks","믿어","신뢰"]): trust += 0.2
    if any(k in low for k in ["싫어","미워","꺼져","hate"]):   trust -= 0.3
    trust = max(0.0, min(1.0, trust))
    closeness = "high" if trust>=0.75 else ("medium" if trust>=0.5 else "low")

    # brief
    def _brief(t): return (t[:120]+"…") if len(t)>140 else t
    lead = " / ".join(f"{m['speaker']}:{_brief(m.get('text',''))}" for m in recent[:6])
    tail = " / ".join(f"{m['speaker']}:{_brief(m.get('text',''))}" for m in recent[-6:])
    summary = f"Earlier: {lead}\nRecent: {tail}"

    return {
        "summary": summary[:max_tokens*4],
        "salient_facts": [],
        "entities": [w for w in keyphrases if re.match(r"[A-Z가-힣][a-z가-힣0-9]+", w)][:12],
        "relationship_state": {"user_trust": round(trust,2), "closeness": closeness},
        "topics": keyphrases[:8]
    }
