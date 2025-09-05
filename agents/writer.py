from typing import List, Dict
from utils.parsing import truncate



SYSTEM_PROMPT = """
You are an expert career coach and technical writer. Draft a concise, evidence-based Korean cover letter.
- Tie the candidate's specific experiences to the JD requirements.
- Use clear, direct sentences. Avoid buzzwords.
- Structure: Intro(2-3), Highlights(3-5 bullets), Closing(2-3).
"""


USER_TEMPLATE = """
[JD]
{jd_text}


[JD Keywords]
{jd_keywords}


[Matched Experiences]
{matches}


Write a Korean draft tailored to the JD. Keep it under 500-700 words.
"""


class CoverLetterWriter:
    def __init__(self, api_key: str = "", model: str = "HuggingFaceH4/zephyr-7b-beta", mode: str = "auto", provider: str = "huggingface"):
        """
        provider: "huggingface" | "openai"
        mode:
        - "auto": 가능하면 LLM 사용, 실패 시 룰기반
        - "llm": 항상 LLM (실패 시 예외)
        - "rule": 항상 룰기반(무료)
        """
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.provider = provider
        
        self._hf = None
        self._openai = None
        if provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                self._hf = InferenceClient(model=model, token=api_key)
            except Exception:
                self._hf = None
        elif provider == "openai":
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=api_key)
            except Exception:
                self._openai = None

    def _rule_based(self, jd_raw: Dict, matches: List[Dict]) -> str:
        jd_text = truncate(jd_raw.get("clean_text", ""), 1200)
        keywords = ", ".join(jd_raw.get("keywords", [])[:18]) or "핵심역량 미추출"
        bullets = "".join([f"- {m['text']}" for m in matches[:5]]) or "- (매칭 결과 없음)"
        draft = f"""안녕하세요. 채용공고에 명시된 역할과 요구역량에 큰 관심이 있어 지원드립니다.


            [핵심 역량 요약]
            - {keywords}


            [공고와 맞닿는 경험]
            {bullets}

            위 경험을 바탕으로 공고의 역할을 빠르게 학습하고, 데이터 기반으로 문제를 정의·검증하며 결과물을 반복 개선하겠습니다.
            검토해 주셔서 감사합니다.
            """
        return draft.strip()
    


    def _hf_call(self, jd_raw: Dict, matches: List[Dict]) -> str:
        if not self._hf:
            raise RuntimeError("Hugging Face InferenceClient 초기화 실패")
        jd_text = truncate(jd_raw.get("clean_text", ""), 3500)
        keywords = ", ".join(jd_raw.get("keywords", [])[:30])
        matched = "".join([f"- {m['text']}" for m in matches])
        prompt = f"{SYSTEM_PROMPT}" + USER_TEMPLATE.format(jd_text=jd_text, jd_keywords=keywords, matches=matched)
        out = self._hf.text_generation(
            prompt,
            max_new_tokens=550,
            temperature=0.5,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False,
        )
        if isinstance(out, dict):
            return (out.get("generated_text") or "").strip()
        return str(out).strip()


    def _openai_call(self, jd_raw: Dict, matches: List[Dict]) -> str:
        if not self._openai:
            raise RuntimeError("OpenAI 클라이언트 초기화 실패")
        jd_text = truncate(jd_raw.get("clean_text", ""), 4000)
        keywords = ", ".join(jd_raw.get("keywords", [])[:30])
        matched = "".join([f"- {m['text']}" for m in matches])
        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(jd_text=jd_text, jd_keywords=keywords, matches=matched)},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()


    def generate_draft(self, jd_raw: Dict, matches: List[Dict]) -> str:
        if self.mode == "rule":
            return self._rule_based(jd_raw, matches)
        try:
            if self.provider == "huggingface" and self._hf:
                return self._hf_call(jd_raw, matches)
            if self.provider == "openai" and self._openai:
                return self._openai_call(jd_raw, matches)
            return self._rule_based(jd_raw, matches)
        except Exception:
            return self._rule_based(jd_raw, matches)