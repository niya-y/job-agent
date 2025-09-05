from typing import List


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class EmbeddingClient:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        if provider == "openai":
            if OpenAI is None:
                raise ImportError("openai SDK가 필요합니다. requirements.txt 참고")
            self.client = OpenAI(api_key=api_key)
        else:
            raise NotImplementedError(f"provider '{provider}'는 아직 지원하지 않습니다.")


    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        raise NotImplementedError