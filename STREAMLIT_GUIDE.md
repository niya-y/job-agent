# 🌐 Streamlit Web App Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_your_token_here
HF_EMBED_MODEL=Snowflake/snowflake-arctic-embed-m
HF_TEXT_MODEL=HuggingFaceH4/zephyr-7b-beta
TOP_K=6
GENERATION_MODE=auto
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📱 Using the Web App

### Step-by-Step Workflow

#### 1️⃣ **Resume Upload**

* Upload your `resume.json` file
* Or use the sample resume from `data/resume.json`

#### 2️⃣ **Job Description**

* Paste the full job posting
* Click "Extract & Analyze JD"
* Skills will be automatically extracted

#### 3️⃣ **Resume Matching**

* Click "Match Resume with JD"
* Your relevant experiences will be retrieved
* View matching scores

#### 4️⃣ **Compatibility Analysis** 🆕

* Click "Analyze Compatibility"
* Get overall score (0-100)
* See strengths and skill gaps
* Get actionable recommendations

#### 5️⃣ **Cover Letter Generation**

* Click "Generate Cover Letter"
* Review and edit the generated letter
* Download as TXT file

---

## ⚙️ Sidebar Settings

### 🔑 API Configuration

* **HuggingFace Token** : Required for matching and generation
* Get token from: https://huggingface.co/settings/tokens
* Make sure to check "Make calls to Inference Providers"

### 🎯 Matching Settings

* **Number of experiences** : How many to retrieve (3-10)

### ✍️ Generation Settings

* **Cover Letter Mode** :
* `auto`: Try LLM → fallback to template
* `huggingface`: Always use LLM
* `rule`: Always use template
* **LLM Model** : Choose between Zephyr-7B or Mistral-7B

---

## 🎨 Features

### ✅ What's Included

* **Real-time JD extraction** with JobSpanBERT
* **Semantic matching** with Snowflake Arctic Embed
* **Compatibility analysis** with detailed insights
* **AI cover letter generation** with Zephyr/Mistral
* **Editable output** - customize before downloading
* **Analysis reports** - JSON export

### 🆕 New in Web Version

* Interactive step-by-step workflow
* Visual compatibility scores
* Real-time feedback
* Download multiple formats
* No command line needed!

---

## 🔧 Troubleshooting

### App won't start

```bash
# Make sure Streamlit is installed
pip install streamlit

# Try running directly
python -m streamlit run app.py
```

### "HF_TOKEN not found" error

* Enter token in sidebar
* Or add to `.env` file
* Or set as environment variable:
  ```bash
  export HF_TOKEN=hf_your_token_herestreamlit run app.py
  ```

### Slow first run

* First time loads models from HuggingFace
* Subsequent runs will be faster
* Models are cached locally

### 403 Forbidden errors

* Check token has "Make calls to Inference Providers" permission
* Try switching to `rule` mode in Generation Settings
* Wait a few minutes (rate limiting)

---

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Add secrets in settings:
   ```
   HF_TOKEN = "hf_your_token"EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"TEXT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
   ```
5. Deploy!

### Deploy to Other Platforms

 **Heroku** :

```bash
# Add Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile
```

 **Docker** :

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 💡 Tips

### For Best Results

1. **Always analyze before generating** - understand your fit first
2. **Review the analysis** - focus on highlighted strengths
3. **Edit the cover letter** - add company-specific details
4. **Test with different JDs** - compare compatibility scores

### Performance Tips

* First run is slower (model loading)
* Keep the app running for multiple JDs
* Use `rule` mode if API is slow
* Cache is your friend!

---

## 📸 Screenshots

### Main Interface

```
┌─────────────────────────────────────┐
│  💼 Job Agent - AI Resume Matcher   │
├─────────────────────────────────────┤
│  1️⃣ Resume Upload                   │
│  2️⃣ Job Description                 │
│  3️⃣ Resume Matching                 │
│  4️⃣ Compatibility Analysis  🆕      │
│  5️⃣ Cover Letter Generation         │
└─────────────────────────────────────┘
```

### Analysis Results

```
Overall Score: 78/100 (Good)
Skill Match: 70%
Experience Relevance: 82/100

💪 Top Strengths        📚 Skill Gaps
✓ python               ✗ kafka
✓ sql                  ✗ kubernetes
✓ apache spark
```

---

## 🆘 Support

Having issues?

1. Check this guide
2. Review [README.md](https://claude.ai/chat/README.md)
3. Check [GitHub issues](https://github.com/YOUR_USERNAME/job-agent/issues)

---

**Happy job hunting! 🎉**
