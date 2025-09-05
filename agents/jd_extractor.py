import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from utils.parsing import html_to_text, normalize_text


STOPWORDS = {"및", "등", "및/또는", "그리고", "또는"}


SKILL_HINTS = [
"python", "sql", "pandas", "numpy", "tensorflow", "pytorch", "spark", "airflow",
"powerbi", "tableau", "aws", "gcp", "azure", "mlops", "docker", "kubernetes",
"llm", "nlp", "genai", "prompt", "faiss", "pinecone", "weaviate"
]


KEY_PATTERNS = [r"자격요건[\s\S]*?(?=우대|주요업무|업무내용|우대사항|혜택|$)", r"주요업무[\s\S]*?(?=자격|우대|혜택|$)"]




def _extract_sections(text: str) -> List[str]:
    sections = []
    for pat in KEY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            sections.append(m.group(0))
    return sections or [text]




def _simple_keywords(text: str, limit: int = 30) -> List[str]:
    text_norm = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z가-힣0-9+#.]+", text_norm)
    # 빈도 기반 + 힌트 스킬 우선
    counts = {}
    for t in tokens:
        if len(t) < 2 or t in STOPWORDS:
            continue
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    base = [w for w, _ in ranked[:limit]]
    # 힌트 스킬을 앞쪽에 보정
    hints = [s for s in SKILL_HINTS if s in counts]
    return list(dict.fromkeys(hints + base))




def extract_jd_from_url(url: str) -> Dict:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.body or soup
    clean = html_to_text(str(main))
    sections = _extract_sections(clean)
    keywords = _simple_keywords("\n".join(sections))
    return {"clean_text": clean, "sections": sections, "keywords": keywords, "source_url": url}




def extract_jd_from_text(text: str) -> Dict:
    clean = normalize_text(text)
    sections = _extract_sections(clean)
    keywords = _simple_keywords("\n".join(sections))
    return {"clean_text": clean, "sections": sections, "keywords": keywords, "source": "manual"}