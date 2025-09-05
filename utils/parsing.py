import re
from typing import List
from bs4 import BeautifulSoup




def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()




def chunk_texts(texts: List[str], chunk_size: int = 400) -> List[str]:
    chunks = []
    for t in texts:
        tokens = re.findall(r".{1,%d}" % chunk_size, t, flags=re.DOTALL)
        chunks.extend([c.strip() for c in tokens if c.strip()])
    return chunks




def truncate(text: str, limit: int) -> str:
    return text[:limit]