# utils/embedding.py
from typing import List

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
            # feature-extraction 엔드포인트 호출
            vecs: List[List[float]] = []
            for t in texts:
                out = self.client.feature_extraction(t)
                # 일부 모델은 [[...]] 형태로 반환 → 1차원으로 정규화
                if out and isinstance(out[0], list):
                    out = out[0]
                vecs.append(out)
            return vecs

        elif self.provider == "local":
            return self.st_model.encode(texts, normalize_embeddings=True).tolist()

        else:
            raise NotImplementedError
