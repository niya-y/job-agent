import os
import json
from typing import List, Dict
import streamlit as st


from agents.jd_extractor import extract_jd_from_url, extract_jd_from_text
from agents.matcher import ResumeMatcher
from agents.writer import CoverLetterWriter
from utils.parsing import normalize_text




# --- Secrets & Config ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

EMBED_PROVIDER = st.secrets.get("EMBED_PROVIDER", os.getenv("EMBED_PROVIDER", "huggingface"))
EMBED_MODEL = st.secrets.get("EMBED_MODEL", os.getenv("EMBED_MODEL", os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")))
TEXT_PROVIDER_DEFAULT = "huggingface" if HF_TOKEN else ("openai" if OPENAI_API_KEY else "rule")
TEXT_MODEL = st.secrets.get("HF_TEXT_MODEL", os.getenv("HF_TEXT_MODEL", "HuggingFaceH4/zephyr-7b-beta"))


TOP_K = int(st.secrets.get("TOP_K", os.getenv("TOP_K", 6)))

st.caption(f"[debug] embed_provider={st.session_state.get('embed_provider', EMBED_PROVIDER)}")

st.set_page_config(page_title="Job Agent", page_icon="💼", layout="wide")
st.title("💼 Job Application Agent")
st.caption("JD → 요건 추출 → 이력서 매칭 → 맞춤 초안 생성")

with st.sidebar:
    st.header("⚙️ 설정")
    st.selectbox("임베딩 프로바이더", options=["huggingface","openai","local"], index=["huggingface","openai","local"].index(EMBED_PROVIDER), key="embed_provider")
    st.text_input("임베딩 모델", value=EMBED_MODEL, key="embed_model")
    st.number_input("Top-K 매칭 개수", min_value=1, max_value=20, value=TOP_K, step=1, key="top_k")


    st.markdown("---")
    st.selectbox("텍스트 생성 프로바이더", options=["huggingface","openai","rule"], index=["huggingface","openai","rule"].index(TEXT_PROVIDER_DEFAULT), key="text_provider")
    st.text_input("텍스트 생성 모델", value=TEXT_MODEL, key="text_model")
    st.radio("Writer 모드", options=["auto","llm","rule"], index=0, key="writer_mode", horizontal=True)


try:
    from utils.embedding import EmbeddingClient
    if st.secrets.get("EMBED_PROVIDER", "huggingface") == "huggingface":
        _warm = EmbeddingClient("huggingface",
                                st.secrets.get("EMBED_MODEL", os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
                                st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "")))
        _ = _warm.embed(["warmup"])
        st.caption("🔄 [warmup] HF 임베딩 준비 완료")
except Exception as _e:
    st.caption(f"⚠️ [warmup 실패] {type(_e).__name__}: {str(_e)[:120]}")

# --- Resume Loader ---
st.subheader("1) 이력서/경험 데이터 업로드")
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("resume.json 업로드", type=["json"])
    if resume_file:
        resume_data = json.loads(resume_file.read().decode("utf-8"))
        st.success("이력서 데이터를 불러왔습니다.")
    else:
    # 샘플 로드
        sample_path = os.path.join("data", "resume.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                resume_data = json.load(f)
            st.info("샘플 resume.json을 사용 중입니다 (data/resume.json).")
        else:
            resume_data = {"summary": "", "experiences": []}
            st.warning("resume.json이 없습니다. 최소 one experience가 필요합니다.")


with col2:
    st.json(resume_data, expanded=False)

# --- JD Input ---
st.subheader("2) 채용공고 입력")
jd_mode = st.radio("입력 방식", ["URL", "텍스트 직접 입력"], horizontal=True)


jd_text = ""
if jd_mode == "URL":
    jd_url = st.text_input("채용공고 URL")
    if st.button("공고 불러오기/추출") and jd_url:
        with st.spinner("JD 추출 중..."):
            jd = extract_jd_from_url(jd_url)
            jd_text = jd.get("clean_text", "")
            st.session_state["jd_keywords"] = jd.get("keywords", [])
            st.session_state["jd_raw"] = jd
        st.success("추출 완료")
else:
    jd_text = st.text_area("공고 텍스트", height=200)
    if st.button("요건 추출") and jd_text.strip():
        with st.spinner("키워드/요건 추출 중..."):
            jd = extract_jd_from_text(jd_text)
            st.session_state["jd_keywords"] = jd.get("keywords", [])
            st.session_state["jd_raw"] = jd
        st.success("추출 완료")


if jd_text:
    st.write("**JD (정제 텍스트)**")
    st.code(jd_text[:3000])

# --- Matching ---
st.subheader("3) 이력서 경험 매칭 (FAISS)")
if st.button("매칭 실행"):
    with st.spinner("임베딩 및 매칭 중..."):
        matcher = ResumeMatcher(
        embed_provider=st.session_state.get("embed_provider", EMBED_PROVIDER),
        embed_model=st.session_state.get("embed_model", EMBED_MODEL),
        api_key=(HF_TOKEN if st.session_state.get("embed_provider", EMBED_PROVIDER)=="huggingface" else OPENAI_API_KEY),
        index_dir="vectorstore"
        )
        matcher.build_or_load_index(resume_data)
        keywords = st.session_state.get("jd_keywords", [])
        if not keywords and jd_text:
        # 키워드가 비어있다면 텍스트에서 간단 추출(백업)
            keywords = normalize_text(jd_text).split()[:50]
        matches = matcher.search(keywords, top_k=st.session_state.get("top_k", TOP_K))
    st.success("매칭 완료")
    st.write("**Top Matches**")
    for m in matches:
        st.markdown(f"- **score**: {m['score']:.3f} — {m['text']}")
    st.session_state["matches"] = matches

# --- Draft Generation ---
st.subheader("4) 맞춤 커버레터/자소서 초안 생성")
if st.button("초안 생성"):
    matches = st.session_state.get("matches", [])
    if not matches:
        st.warning("먼저 매칭을 실행하세요.")
    else:
        with st.spinner("초안 생성 중..."):
            writer = CoverLetterWriter(
                api_key=(HF_TOKEN if st.session_state.get("text_provider", TEXT_PROVIDER_DEFAULT)=="huggingface" else OPENAI_API_KEY),
                model=st.session_state.get("text_model", TEXT_MODEL),
                mode=st.session_state.get("writer_mode", "auto"),
                provider=st.session_state.get("text_provider", TEXT_PROVIDER_DEFAULT),
            )
            jd_raw = st.session_state.get("jd_raw", {"clean_text": jd_text, "keywords": []})
            draft = writer.generate_draft(jd_raw, matches)
        st.success("생성 완료")
        st.markdown("### ✍️ 생성된 초안")
        st.write(draft)
        st.download_button("⬇️ 초안 .md 다운로드", data=draft, file_name="cover_letter.md")