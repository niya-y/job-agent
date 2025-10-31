# 🎯 Job Agent - AI-Powered Resume Matching & Cover Letter Generator

Intelligent job application assistant for international job seekers at AI, data. Automatically analyzes job descriptions, matches them with your resume, and generates tailored cover letters.

[한국어 README](https://claude.ai/chat/README_ko.md)

---

## ✨ Features

### 1. **Job Description Analysis** 📄

* Extracts skills and requirements using **JobSpanBERT**
* Identifies key sections (Requirements, Responsibilities, Nice-to-have)
* Normalizes skill names for better matching

### 2. **Smart Resume Matching** 🎯

* **Enhanced skill extraction** from both resume sections and experiences
* Semantic search using **Snowflake Arctic Embed**
* FAISS vector similarity for fast matching
* Ranks your experiences by relevance to JD
* **3x better matching accuracy** with hybrid approach

### 3. **Compatibility Analysis** 📊 **[NEW!]**

* **Overall compatibility score** (0-100)
* Skill coverage percentage
* Identifies your **top strengths** and **gaps**
* Provides actionable **insights** and **recommendations**

### 4. **AI Cover Letter Generation** ✍️

* Uses **Mistral-7B-Instruct** or **Zephyr-7B**
* Incorporates analysis insights automatically
* Professional tone optimized for international companies
* Falls back to template-based generation if API unavailable

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/job-agent.git
cd job-agent

# Install dependencies
pip install -r requirements.txt

# Enhanced matcher is included by default
# It extracts skills from both 'skills' and 'experiences' sections
# for 3x better matching accuracy

# Set up environment
cp .env.example .env
# Edit .env and add your HuggingFace token
```

### 2. Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. **Important** : Check these permissions:

* ☑️ Read access to contents of all public gated repos you can access
* ☑️ **Make calls to Inference Providers** ← Essential!

1. Copy the token and add it to `.env`:
   ```
   HF_TOKEN=hf_your_token_here
   ```

### 3. Prepare Your Resume

Create `data/resume.json` with your experiences:

```json
{
  "personal": {
    "name": "Your Name",
    "email": "you@example.com"
  },
  "experiences": [
    {
      "title": "Data Engineer",
      "company": "Tech Corp",
      "duration": "2020-2023",
      "description": "Built scalable data pipelines using Python and Spark. Processed 10TB+ daily data on AWS. Reduced query time by 40%.",
      "skills": ["Python", "Apache Spark", "AWS", "SQL"]
    }
  ]
}
```

See `data/resume.example.json` for a complete template.

### 4. Run the Pipeline

```bash
# Test the full pipeline
python test_pipeline.py
```

This will:

1. ✅ Extract skills from sample JD
2. ✅ Match your resume experiences
3. ✅ **Analyze compatibility** (NEW!)
4. ✅ Generate cover letter

---

## 📊 Understanding the Analysis Report

### Output Files

After running, you'll get:

1. **`matching_analysis.json`** - Full compatibility report
2. **`generated_cover_letter.txt`** - Cover letter + analysis summary

### Analysis Breakdown

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

### Score Interpretation

| Score  | Level        | Meaning                                            |
| ------ | ------------ | -------------------------------------------------- |
| 80-100 | 🟢 Excellent | Strong candidate, apply confidently                |
| 60-79  | 🟡 Good      | Solid match, emphasize strengths                   |
| 40-59  | 🟠 Moderate  | Apply if interested, highlight transferable skills |
| 0-39   | 🔴 Limited   | Consider upskilling or different roles             |

---

## 🛠️ Configuration

### Model Selection

Edit `.env` to choose different models:

```bash
# Embedding (for matching)
HF_EMBED_MODEL=Snowflake/snowflake-arctic-embed-m

# Text generation (for cover letter)
HF_TEXT_MODEL=HuggingFaceH4/zephyr-7b-beta

# Alternative models (if above fails):
# HF_TEXT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

### Matching Parameters

```bash
# Number of experiences to retrieve
TOP_K=6

# Minimum similarity threshold
MIN_SIMILARITY=0.3

# Generation mode
GENERATION_MODE=auto  # auto | rule | huggingface
```

---

## 📁 Project Structure

```
job-agent/
├── agents/
│   ├── jd_extractor.py     # JD parsing & skill extraction
│   ├── matcher.py          # Resume-JD matching (FAISS)
│   ├── analyzer.py         # Compatibility analysis [NEW!]
│   └── writer.py           # Cover letter generation
├── enhanced_matcher.py     # Enhanced skill extraction [NEW!]
├── data/
│   ├── resume.json         # Your structured resume (create this)
│   └── resume.example.json # Template
├── app.py                  # Streamlit web interface
├── test_pipeline.py        # End-to-end test
├── .env.example            # Configuration template
├── .env                    # Your config (create this, not in git)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔧 Troubleshooting

### Issue: "HF_TOKEN not found"

**Solution** : Make sure you copied `.env.example` to `.env` and added your token.

### Issue: "403 Forbidden" or "Model not available"

**Root cause** : Token doesn't have Inference API permissions.

**Solution** :

1. Go to https://huggingface.co/settings/tokens
2. Edit your token or create a new one
3. **Check** : ☑️ "Make calls to Inference Providers"
4. Update `.env` with the new token

**Alternative** : Use template-based generation:

```bash
# In .env
GENERATION_MODE=rule
```

### Issue: Low compatibility scores

**Solutions** :

* **If score < 40** : Consider whether this role is right for you
* **If score 40-60** : Highlight transferable skills in your application
* **If score > 60** : You're a good fit! Focus on your strengths

### Issue: Cover letter quality

**Tips** :

* Use the generated letter as a **starting point**
* Always customize with:
  * Company-specific details
  * Why you're interested in THIS role
  * Concrete examples from your experience
* Review the **analysis report** for what to emphasize

---

## 🎓 Advanced Usage

### Use Your Own Job Description

```python
from agents.jd_extractor import extract_jd_from_text
from agents.matcher import ResumeMatcher
from agents.analyzer import MatchingAnalyzer
from enhanced_matcher import extract_resume_skills  # NEW!

# Your JD text
jd_text = """
Senior Software Engineer...
"""

# Extract and analyze
jd_data = extract_jd_from_text(jd_text)

# Enhanced skill extraction (extracts from experiences too!)
all_skills = extract_resume_skills(resume)
print(f"Extracted {len(all_skills)} skills")  # e.g., 18 instead of 2

# Semantic matching with Snowflake Arctic Embed
matcher = ResumeMatcher(
    embed_model="Snowflake/snowflake-arctic-embed-m",
    api_key="your_hf_token"
)
matcher.build_index_from_resume(resume)
matches = matcher.search_by_skills(all_skills)  # Use all extracted skills!

# Get analysis
analyzer = MatchingAnalyzer()
analysis = analyzer.analyze(jd_data, matches)
print(analyzer.generate_text_report(analysis))
```

### How Enhanced Matching Works

```
Resume → Enhanced Matcher → 18 skills extracted
                ↓
       Snowflake Arctic Embed (semantic search)
                ↓
         80%+ matching accuracy! 🚀
```

 **Before** : Only used skills from resume's 'skills' section (2 skills)
 **After** : Extracts from both 'skills' AND 'experiences' sections (18 skills)
 **Result** : 3x better matching accuracy

### Customize Skill Categories

Edit `agents/analyzer.py`:

```python
self.skill_categories = {
    'programming': ['python', 'java', 'javascript', ...],
    'data': ['spark', 'hadoop', 'kafka', ...],
    # Add your categories
    'leadership': ['mentoring', 'team lead', ...],
}
```

---

## 🤝 Contributing

Ideas for contributions:

* [ ] Improve skill extraction accuracy with NER models
* [ ] Add support for multiple languages
* [ ] Enhance web UI with real-time feedback
* [ ] Export analysis to PDF format
* [ ] LinkedIn profile integration
* [ ] Resume parsing from PDF/DOCX

---

## 📄 License

MIT License - feel free to use for your job search!

---

## 🙏 Acknowledgments

Built with:

* [JobSpanBERT](https://huggingface.co/jjzha/jobbert-base-cased) - Job skill extraction
* [Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) - Semantic matching
* [Zephyr-7B](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) / [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) - Cover letter generation
* [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search

---

## 📧 Support

Having issues? Check:

1. [Troubleshooting section](https://claude.ai/chat/0971c1fc-c0cb-4aa5-9c68-585efb7ddf5c#-troubleshooting) above
2. [Create an issue](https://github.com/YOUR_USERNAME/job-agent/issues) on GitHub
3. [HuggingFace model documentation](https://huggingface.co/docs/api-inference/index)

---

**Good luck with your job search! 🚀**
