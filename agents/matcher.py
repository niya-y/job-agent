import os
from typing import List, Dict


from utils.embedding import EmbeddingClient
from utils.parsing import chunk_texts


try:
    import faiss # type: ignore
except Exception as e:
    faiss = None

class ResumeMatcher:
    def __init__(self, embed_provider: str, embed_model: str, api_key: str, index_dir: str = "vectorstore"):
        self.client = EmbeddingClient(provider=embed_provider, model=embed_model, api_key=api_key)
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "meta.json")
        os.makedirs(index_dir, exist_ok=True)
        self.index = None
        self.meta = []


    def build_or_load_index(self, resume_data: Dict):
        """Build FAISS index from resume.json (experiences list)."""
        experiences = []
        for exp in resume_data.get("experiences", []):
            txt = exp.get("description", "")
            if exp.get("title"): txt = f"{exp['title']} — {txt}"
            if exp.get("company"): txt = f"{exp['company']} | {txt}"
            experiences.append(txt)
        if not experiences:
            raise ValueError("resume.json에 experiences가 없습니다.")
        chunks = chunk_texts(experiences, chunk_size=400)
        embeddings = self.client.embed(chunks)
        dim = len(embeddings[0])
        if faiss is None:
            raise ImportError("faiss-cpu가 설치되어야 합니다. requirements.txt 참고")
        index = faiss.IndexFlatIP(dim)
        import numpy as np
        vecs = np.array(embeddings, dtype="float32")
        # 정규화로 내적=코사인 유사도 효과
        faiss.normalize_L2(vecs)
        index.add(vecs)
        self.index = index
        self.meta = chunks
        # 저장
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            import json
            json.dump(self.meta, f, ensure_ascii=False, indent=2)


    def search(self, keywords: List[str], top_k: int = 6) -> List[Dict]:
        if self.index is None:
        # Try load
            try:
                import json
                import numpy as np
                import faiss
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception as e:
                raise RuntimeError("인덱스를 먼저 생성하세요(build_or_load_index).")
        # 질의 임베딩
        query = " ".join(keywords[:50])
        q_emb = self.client.embed([query])[0]
        import numpy as np
        q = np.array([q_emb], dtype="float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            results.append({"score": float(score), "text": self.meta[idx]})
        return results