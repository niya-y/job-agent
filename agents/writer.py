# agents/writer.py
from typing import List, Dict
from utils.parsing import truncate

SYSTEM_PROMPT = """You are an expert career coach and technical writer..."""
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
    def __init__(self, api_key: str = "", model: str = "HuggingFaceH4/zephyr-7b-beta",
                 mode: str = "auto", provider: str = "huggingface"):
        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.provider = provider
        self._hf = None
        self._openai = None

        if provider == "huggingface":
            from huggingface_hub import InferenceClient
            self._hf = InferenceClient(model=model, token=api_key)
        elif provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(api_key=api_key)

    def _rule_based(self, jd_raw: Dict, matches: List[Dict]) -> str:
        jd_text = truncate(jd_raw.get("clean_text", ""), 1200)
        keywords = ", ".join(jd_raw.get("keywords", [])[:18]) or "핵심역량 미추출"
        bullets = "\n".join([f"- {m['text']}" for m in matches[:5]]) or "- (매칭 결과 없음)"
        return f"""안녕하세요...
[핵심 역량 요약]
- {keywords}

[제가 공고와 맞닿는 경험]
{bullets}

감사합니다.
""".strip()

    def _hf_call(self, jd_raw: Dict, matches: List[Dict]) -> str:
        jd_text = truncate(jd_raw.get("clean_text", ""), 3500)
        keywords = ", ".join(jd_raw.get("keywords", [])[:30])
        matched = "\n".join([f"- {m['text']}" for m in matches])
        prompt = f"{SYSTEM_PROMPT}\n\n" + USER_TEMPLATE.format(
            jd_text=jd_text, jd_keywords=keywords, matches=matched
        )
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
        jd_text = truncate(jd_raw.get("clean_text", ""), 4000)
        keywords = ", ".join(jd_raw.get("keywords", [])[:30])
        matched = "\n".join([f"- {m['text']}" for m in matches])
        from openai import OpenAI
        client = self._openai or OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": USER_TEMPLATE.format(jd_text=jd_text, jd_keywords=keywords, matches=matched)},
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
