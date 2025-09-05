# Job Application Agent (Streamlit)


JD → 핵심요건 추출 → 이력서/경험 매칭(FAISS) → 맞춤 커버레터/자기소개서 초안 생성.


## 🔧 로컬 실행
```bash
python -m venv .venv && source .venv/bin/activate # (Windows는 .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env # 또는 Streamlit Secrets 사용
streamlit run app.py