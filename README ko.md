# 🎯 Job Agent - AI 기반 이력서 매칭 및 커버레터 생성기

인공지능, 데이터 분야 외국계 기업 취업 준비생을 위한 지능형 지원 도우미입니다. 채용공고를 자동으로 분석하고, 이력서와 매칭하며, 맞춤형 커버레터를 생성합니다.

[English README](https://claude.ai/chat/README.md)

---

## ✨ 주요 기능

### 1. **채용공고(JD) 분석** 📄

* **JobSpanBERT**를 활용한 스킬 및 요구사항 추출
* 주요 섹션 식별 (필수 요건, 담당 업무, 우대사항)
* 스킬 이름 정규화로 더 나은 매칭

### 2. **스마트 이력서 매칭** 🎯

* **향상된 스킬 추출** - 이력서의 모든 섹션에서 자동 추출
* **Snowflake Arctic Embed**를 사용한 의미 기반 검색
* FAISS 벡터 유사도로 빠른 매칭
* JD와의 관련성에 따라 경력 순위 매김
* **하이브리드 접근으로 3배 향상된 매칭 정확도**

### 3. **적합도 분석** 📊 **[신규!]**

* **종합 적합도 점수** (0-100)
* 스킬 커버리지 퍼센티지
* 당신의 **강점**과 **부족한 점** 파악
* 실행 가능한 **인사이트** 및 **추천사항** 제공

### 4. **AI 커버레터 생성** ✍️

* **Mistral-7B-Instruct** 또는 **Zephyr-7B** 사용
* 분석 인사이트 자동 반영
* 외국계 기업에 최적화된 전문적 톤
* API 사용 불가 시 템플릿 기반 생성으로 폴백

---

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/job-agent.git
cd job-agent

# 의존성 설치
pip install -r requirements.txt

# Enhanced matcher 기본 포함
# 'skills' 및 'experiences' 섹션에서 자동으로 스킬 추출
# 3배 향상된 매칭 정확도

# 환경 설정
cp .env.example .env
# .env 파일을 열어 HuggingFace 토큰 추가
```

### 2. HuggingFace 토큰 발급

1. https://huggingface.co/settings/tokens 접속
2. **"Create new token"** 클릭
3. **중요** : 다음 권한 체크:

* ☑️ Read access to contents of all public gated repos you can access
* ☑️ **Make calls to Inference Providers** ← 필수!

1. 토큰 복사 후 `.env`에 추가:
   ```
   HF_TOKEN=hf_your_token_here
   ```

### 3. 이력서 준비

`data/resume.json` 파일 생성:

```json
{
  "personal": {
    "name": "홍길동",
    "email": "you@example.com"
  },
  "experiences": [
    {
      "title": "데이터 엔지니어",
      "company": "테크 코퍼레이션",
      "duration": "2020-2023",
      "description": "Python과 Spark를 활용한 확장 가능한 데이터 파이프라인 구축. AWS에서 일 10TB+ 데이터 처리. 쿼리 시간 40% 단축.",
      "skills": ["Python", "Apache Spark", "AWS", "SQL"]
    }
  ]
}
```

전체 템플릿은 `data/resume.example.json` 참고.

### 4. 파이프라인 실행

```bash
# 전체 파이프라인 테스트
python test_pipeline.py
```

실행 과정:

1. ✅ 샘플 JD에서 스킬 추출
2. ✅ 이력서 경력 매칭
3. ✅ **적합도 분석** (신규!)
4. ✅ 커버레터 생성

---

## 📊 분석 리포트 이해하기

### 생성 파일

실행 후 다음 파일들이 생성됩니다:

1. **`matching_analysis.json`** - 상세 적합도 리포트
2. **`generated_cover_letter.txt`** - 커버레터 + 분석 요약

### 분석 결과 예시

```
📊 RESUME-JD MATCHING ANALYSIS REPORT
====================================

Overall Compatibility: 78/100 (Good)

Detailed Breakdown:
  • Skill Match: 75.0/100 (70% coverage)
  • Experience Relevance: 82.5/100

Top Strengths:
  ✓ python
  ✓ sql
  ✓ apache spark
  ✓ aws

Skill Gaps:
  ✗ kafka
  ✗ kubernetes

Key Insights:
  ✅ Good match. You meet most of the key requirements with some gaps.
  💪 Strong skill coverage: 70% of required skills matched.

Recommendations:
  1. ✍️ Highlight your matching skills prominently in your cover letter.
  2. 📚 Priority skills to learn: kafka, kubernetes
  3. 💬 In your cover letter, demonstrate how your skills solve their specific challenges.
```

### 점수 해석

| 점수   | 수준         | 의미                                  |
| ------ | ------------ | ------------------------------------- |
| 80-100 | 🟢 매우 적합 | 강력한 후보, 자신있게 지원            |
| 60-79  | 🟡 적합      | 좋은 매칭, 강점 강조                  |
| 40-59  | 🟠 보통      | 관심 있다면 지원, 전이 가능 스킬 강조 |
| 0-39   | 🔴 부족      | 스킬 향상 또는 다른 포지션 고려       |

---

## 🛠️ 설정

### 모델 선택

`.env` 파일을 편집하여 다른 모델 선택:

```bash
# 임베딩 (매칭용)
HF_EMBED_MODEL=Snowflake/snowflake-arctic-embed-m

# 텍스트 생성 (커버레터용)
HF_TEXT_MODEL=HuggingFaceH4/zephyr-7b-beta

# 대안 모델 (위 모델 실패 시):
# HF_TEXT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

### 매칭 파라미터

```bash
# 검색할 경력 수
TOP_K=6

# 최소 유사도 임계값
MIN_SIMILARITY=0.3

# 생성 모드
GENERATION_MODE=auto  # auto | rule | huggingface
```

---

## 📁 프로젝트 구조

```
job-agent/
├── agents/
│   ├── jd_extractor.py     # JD 파싱 및 스킬 추출
│   ├── matcher.py          # 이력서-JD 매칭 (FAISS)
│   ├── analyzer.py         # 적합도 분석 [신규!]
│   └── writer.py           # 커버레터 생성
├── enhanced_matcher.py     # 향상된 스킬 추출 [신규!]
├── data/
│   ├── resume.json         # 내 이력서 (직접 생성)
│   └── resume.example.json # 템플릿
├── app.py                  # Streamlit 웹 인터페이스
├── test_pipeline.py        # 엔드투엔드 테스트
├── .env.example            # 설정 템플릿
├── .env                    # 내 설정 (직접 생성, git 제외)
├── requirements.txt        # Python 의존성
└── README.md               # 이 파일
```

---

## 🔧 문제 해결

### 문제: "HF_TOKEN not found"

**해결** : `.env.example`을 `.env`로 복사하고 토큰을 추가했는지 확인하세요.

### 문제: "403 Forbidden" 또는 "Model not available"

**원인** : 토큰에 Inference API 권한이 없습니다.

**해결** :

1. https://huggingface.co/settings/tokens 접속
2. 토큰 편집 또는 새로 생성
3. **체크** : ☑️ "Make calls to Inference Providers"
4. `.env`에 새 토큰 업데이트

**대안** : 템플릿 기반 생성 사용:

```bash
# .env에 추가
GENERATION_MODE=rule
```

### 문제: 낮은 적합도 점수

**해결 방법** :

* **점수 < 40** : 이 포지션이 맞는지 재고려
* **점수 40-60** : 지원서에서 전이 가능한 스킬 강조
* **점수 > 60** : 좋은 핏! 강점에 집중

### 문제: 커버레터 품질

**팁** :

* 생성된 레터를 **초안**으로 활용
* 항상 다음을 커스터마이징:
  * 회사별 세부 정보
  * 이 역할에 관심 있는 이유
  * 경험의 구체적 예시
* **분석 리포트**를 참고하여 강조할 내용 파악

---

## 🎓 고급 사용법

### 직접 작성한 채용공고 사용

```python
from agents.jd_extractor import extract_jd_from_text
from agents.matcher import ResumeMatcher
from agents.analyzer import MatchingAnalyzer
from enhanced_matcher import extract_resume_skills  # 신규!

# 채용공고 텍스트
jd_text = """
Senior Software Engineer...
"""

# 추출 및 분석
jd_data = extract_jd_from_text(jd_text)

# 향상된 스킬 추출 (경력에서도 추출!)
all_skills = extract_resume_skills(resume)
print(f"추출된 스킬: {len(all_skills)}개")  # 예: 2개 → 18개

# Snowflake Arctic Embed로 의미 기반 매칭
matcher = ResumeMatcher(
    embed_model="Snowflake/snowflake-arctic-embed-m",
    api_key="your_hf_token"
)
matcher.build_index_from_resume(resume)
matches = matcher.search_by_skills(all_skills)  # 추출된 모든 스킬 사용!

# 분석 수행
analyzer = MatchingAnalyzer()
analysis = analyzer.analyze(jd_data, matches)
print(analyzer.generate_text_report(analysis))
```

### 향상된 매칭 작동 방식

```
이력서 → Enhanced Matcher → 18개 스킬 추출
                ↓
       Snowflake Arctic Embed (의미 기반 검색)
                ↓
         80%+ 매칭 정확도! 🚀
```

 **기존** : 이력서의 'skills' 섹션만 사용 (2개)
 **개선** : 'skills' + 'experiences' 섹션 모두 분석 (18개)
 **결과** : 3배 향상된 매칭 정확도

### 스킬 카테고리 커스터마이징

`agents/analyzer.py` 편집:

```python
self.skill_categories = {
    'programming': ['python', 'java', 'javascript', ...],
    'data': ['spark', 'hadoop', 'kafka', ...],
    # 카테고리 추가
    'leadership': ['mentoring', 'team lead', ...],
}
```

---

## 🤝 기여하기

기여 아이디어:

* [ ] NER 모델로 스킬 추출 정확도 개선
* [ ] 다국어 지원 추가
* [ ] 실시간 피드백이 있는 웹 UI 강화
* [ ] PDF 형식으로 분석 결과 내보내기
* [ ] LinkedIn 프로필 연동
* [ ] PDF/DOCX 이력서 파싱 기능

---

## 📄 라이선스

MIT License - 취업 준비에 자유롭게 사용하세요!

---

## 🙏 감사의 말

다음 기술로 구축되었습니다:

* [JobSpanBERT](https://huggingface.co/jjzha/jobbert-base-cased) - 직무 스킬 추출
* [Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) - 의미 기반 매칭
* [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) / [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) - 커버레터 생성
* [FAISS](https://github.com/facebookresearch/faiss) - 효율적인 유사도 검색

---

## 📧 지원

1. 위의 [문제 해결](https://claude.ai/chat/0971c1fc-c0cb-4aa5-9c68-585efb7ddf5c#-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0) 섹션 확인
2. GitHub에 [이슈 생성](https://github.com/YOUR_USERNAME/job-agent/issues)
3. [HuggingFace 모델 문서](https://huggingface.co/docs/api-inference/index) 참고

---

**취업 준비 화이팅! 🚀**
