# utils/embedding.py
from typing import List
import time

def _to_1d_vector(x):
    """List/np.array/토큰별 2D도 모두 1D로 변환 (평균 풀링)."""
    try:
        import numpy as np
        arr = np.asarray(x, dtype="float32")
        # shape: (tokens, dim) or (dim,)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        return arr.astype("float32").tolist()
    except Exception:
        # numpy가 없거나 예외인 경우: 최대한 리스트로 변환 후 1차원화
        if isinstance(x, list) and x and isinstance(x[0], list):
            # [[token_emb], ...] 형태면 평균
            dim = len(x[0])
            cols = [sum(row[i] for row in x) / len(x) for i in range(dim)]
            return cols
        return x  # 이미 1D 리스트라고 가정

class EmbeddingClient:
    def __init__(self, provider: str, model: str, api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

        elif provider == "huggingface":
            # Hugging Face Inference API
            from huggingface_hub import InferenceClient
            # sentence-transformers 계열 모델명 예: "sentence-transformers/all-MiniLM-L6-v2"
            self.client = InferenceClient(model=model, token=api_key)

        elif provider == "local":
            # 로컬 임베딩(옵션)
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model or "all-MiniLM-L6-v2")

        else:
            raise NotImplementedError(f"provider '{provider}'는 아직 지원하지 않습니다.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        
        elif self.provider == "huggingface":
            # ❗ 핵심: 한 번에 리스트로 보내고, 모델 준비될 때까지 기다리고, 재시도
            max_retries = 3
            backoff = 2.0
            last_err = None

            # 너무 긴 문장은 잘라서 타임아웃 방지 (필요시 조정)
            clipped = [t[:2000] for t in texts]

            for attempt in range(1, max_retries + 1):
                try:
                    out = self.client.feature_extraction(
                        clipped,
                        wait_for_model=True,  # 콜드 스타트 기다리기
                    )
                     # 반환: List[vector] 또는 List[token_vectors]
                    # 각 항목을 1D 벡터로 통일
                    return [_to_1d_vector(item) for item in out]
                    
                except Exception as e:
                    last_err = e
                    if attempt == max_retries:
                        raise
                    time.sleep(backoff)
                    backoff *= 2.0

        
        elif self.provider == "local":
            return self.st_model.encode(texts, normalize_embeddings=True).tolist()

        else:
            raise NotImplementedError
