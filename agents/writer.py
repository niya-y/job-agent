from typing import List, Dict
from utils.parsing import truncate


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


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
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("openai SDK가 필요합니다. requirements.txt 참고")
        self.client = OpenAI(api_key=api_key)
        self.model = model


    def generate_draft(self, jd_raw: Dict, matches: List[Dict]) -> str:
        jd_text = truncate(jd_raw.get("clean_text", ""), 4000)
        keywords = ", ".join(jd_raw.get("keywords", [])[:30])
        matched = "\n".join([f"- {m['text']}" for m in matches])
        user = USER_TEMPLATE.format(jd_text=jd_text, jd_keywords=keywords, matches=matched)


        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()