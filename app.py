"""
Streamlit Web Application for Job Agent
AI-powered resume matching and cover letter generation
"""

import os
import json
from typing import Dict, List, Optional
import streamlit as st

from agents.jd_extractor_v2 import extract_jd_from_text
from agents.matcher import ResumeMatcher
from agents.analyzer import MatchingAnalyzer
from agents.writer import CoverLetterWriter

# Import enhanced matcher for improved skill extraction
try:
    from enhanced_matcher import (
        extract_resume_skills,
        match_resume_to_jd_skills,
        EnhancedMatcher
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Job Agent - AI Resume Matcher",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
if 'matches' not in st.session_state:
    st.session_state.matches = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'cover_letter' not in st.session_state:
    st.session_state.cover_letter = None
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None


# ============================================
# CONFIGURATION & SECRETS
# ============================================

def load_config() -> Dict:
    """Load configuration from secrets or environment variables"""
    # Try to load from secrets, fallback to environment variables
    try:
        return {
            'hf_token': st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", "")),
            'embed_model': st.secrets.get("EMBED_MODEL", os.getenv("HF_EMBED_MODEL", "Snowflake/snowflake-arctic-embed-m")),
            'text_model': st.secrets.get("TEXT_MODEL", os.getenv("HF_TEXT_MODEL", "HuggingFaceH4/zephyr-7b-beta")),
            'top_k': int(st.secrets.get("TOP_K", os.getenv("TOP_K", 6))),
            'generation_mode': st.secrets.get("GENERATION_MODE", os.getenv("GENERATION_MODE", "auto"))
        }
    except Exception:
        # Secrets not available, use environment variables only
        return {
            'hf_token': os.getenv("HF_TOKEN", ""),
            'embed_model': os.getenv("HF_EMBED_MODEL", "Snowflake/snowflake-arctic-embed-m"),
            'text_model': os.getenv("HF_TEXT_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
            'top_k': int(os.getenv("TOP_K", 6)),
            'generation_mode': os.getenv("GENERATION_MODE", "auto")
        }

config = load_config()


# ============================================
# HEADER
# ============================================

st.title("💼 Job Agent - AI Resume Matcher")
st.markdown("""
**Intelligent job application assistant** - Analyzes job descriptions, matches with your resume, 
and generates tailored cover letters with compatibility analysis.
""")

st.markdown("---")


# ============================================
# SIDEBAR - SETTINGS
# ============================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Enhanced matcher status
    if ENHANCED_AVAILABLE:
        st.success("✨ Enhanced Matcher: Active")
        st.caption("3x better skill extraction")
    else:
        st.info("ℹ️ Basic Matcher: Active")
        st.caption("Add enhanced_matcher.py for better results")
    
    st.markdown("---")
    
    st.subheader("🔑 API Configuration")
    hf_token_input = st.text_input(
        "HuggingFace Token",
        value=config['hf_token'],
        type="password",
        help="Get token from https://huggingface.co/settings/tokens"
    )
    
    if hf_token_input:
        config['hf_token'] = hf_token_input
        st.success("✓ Token configured")
    else:
        st.warning("⚠️ Please enter HuggingFace token")
    
    st.markdown("---")
    
    st.subheader("🎯 Matching Settings")
    config['top_k'] = st.slider(
        "Number of experiences to match",
        min_value=3,
        max_value=10,
        value=config['top_k'],
        help="How many relevant experiences to retrieve"
    )
    
    st.markdown("---")
    
    st.subheader("✍️ Generation Settings")
    config['generation_mode'] = st.selectbox(
        "Cover Letter Mode",
        options=["auto", "huggingface", "rule"],
        index=["auto", "huggingface", "rule"].index(config['generation_mode']),
        help="auto: try LLM → fallback to template | rule: always use template"
    )
    
    config['text_model'] = st.selectbox(
        "LLM Model",
        options=[
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        index=0,
        help="Model for generating cover letters"
    )
    
    st.markdown("---")
    
    with st.expander("🔧 Advanced Settings"):
        config['embed_model'] = st.text_input(
            "Embedding Model",
            value=config['embed_model'],
            help="Model for semantic matching"
        )
        
        st.caption("Current models:")
        st.code(f"Embed: {config['embed_model']}\nText: {config['text_model']}")


# ============================================
# STEP 1: RESUME UPLOAD
# ============================================

st.header("1️⃣ Resume Upload")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume.json",
        type=['json'],
        help="Upload a structured resume in JSON format"
    )
    
    if uploaded_file:
        try:
            resume_data = json.load(uploaded_file)
            st.session_state.resume_data = resume_data
            st.success(f"✓ Resume loaded: {len(resume_data.get('experiences', []))} experiences")
        except Exception as e:
            st.error(f"❌ Error loading resume: {str(e)}")
    
    # Load sample if no upload
    if not st.session_state.resume_data:
        sample_path = "data/resume.json"
        if os.path.exists(sample_path):
            try:
                with open(sample_path, 'r', encoding='utf-8') as f:
                    st.session_state.resume_data = json.load(f)
                st.info("ℹ️ Using sample resume from data/resume.json")
            except Exception as e:
                st.warning(f"⚠️ Could not load sample: {str(e)}")
        else:
            st.warning("⚠️ No resume loaded. Please upload resume.json")

with col2:
    if st.session_state.resume_data:
        with st.expander("📄 View Resume Data"):
            st.json(st.session_state.resume_data)


# ============================================
# STEP 2: JOB DESCRIPTION INPUT
# ============================================

st.header("2️⃣ Job Description")

jd_text = st.text_area(
    "Paste the job description here",
    height=300,
    placeholder="Copy and paste the full job posting...",
    help="Include requirements, responsibilities, and qualifications"
)

if st.button("🔍 Extract & Analyze JD", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.error("❌ Please enter a job description")
    else:
        with st.spinner("Extracting skills and requirements..."):
            try:
                jd_data = extract_jd_from_text(jd_text)
                st.session_state.jd_data = jd_data
                
                if jd_data['success']:
                    st.success(f"✓ Extracted {len(jd_data['skills'])} skills from JD")
                    
                    # Display extracted skills
                    with st.expander("📊 Extracted Skills", expanded=True):
                        skills_text = ", ".join(jd_data['skills'][:20])
                        if len(jd_data['skills']) > 20:
                            skills_text += f" ... and {len(jd_data['skills']) - 20} more"
                        st.write(skills_text)
                else:
                    st.warning("⚠️ Extraction had issues, but will proceed with basic analysis")
                    
            except Exception as e:
                st.error(f"❌ Error extracting JD: {str(e)}")


# ============================================
# STEP 3: RESUME MATCHING
# ============================================

st.header("3️⃣ Resume Matching & Analysis")

if st.button("🎯 Match Resume with JD", type="primary", use_container_width=True):
    if not st.session_state.jd_data:
        st.error("❌ Please extract JD first (Step 2)")
    elif not st.session_state.resume_data:
        st.error("❌ Please upload resume first (Step 1)")
    elif not config['hf_token']:
        st.error("❌ Please enter HuggingFace token in sidebar")
    else:
        with st.spinner("Matching your experiences with job requirements..."):
            try:
                # Enhanced skill extraction if available
                if ENHANCED_AVAILABLE:
                    with st.spinner("Extracting all skills from resume..."):
                        all_skills = extract_resume_skills(st.session_state.resume_data)
                        explicit_skills = st.session_state.resume_data.get('skills', [])
                        
                        st.info(f"✨ Enhanced extraction: {len(explicit_skills)} → {len(all_skills)} skills")
                        
                        # Get direct skill match
                        skill_match = match_resume_to_jd_skills(
                            st.session_state.resume_data,
                            st.session_state.jd_data['skills']
                        )
                        
                        # Store for later use
                        st.session_state.resume_data['_skill_match'] = skill_match
                        
                        # Use extracted skills
                        search_skills = all_skills
                else:
                    search_skills = st.session_state.jd_data['skills']
                
                # Ensure search_skills is a list
                if not isinstance(search_skills, list):
                    search_skills = list(search_skills)
                
                # Initialize matcher with Snowflake Arctic Embed
                matcher = ResumeMatcher(
                    embed_model=config['embed_model'],
                    api_key=config['hf_token']
                )
                
                # Build index and search
                matcher.build_index_from_resume(st.session_state.resume_data)
                matches = matcher.search_by_skills(
                    search_skills,
                    top_k=config['top_k']
                )
                
                st.session_state.matches = matches
                
                # Success message with skill match info
                if ENHANCED_AVAILABLE and '_skill_match' in st.session_state.resume_data:
                    skill_match = st.session_state.resume_data['_skill_match']
                    st.success(f"✓ Found {len(matches)} relevant experiences | Skill match: {skill_match['match_percentage']:.0f}%")
                else:
                    st.success(f"✓ Found {len(matches)} relevant experiences")
                
            except Exception as e:
                st.error(f"❌ Error during matching: {str(e)}")
                st.session_state.matches = None

# Display matches
if st.session_state.matches:
    st.subheader("📋 Top Matching Experiences")
    
    for i, match in enumerate(st.session_state.matches, 1):
        with st.expander(f"Match {i} - Score: {match['score']:.3f}"):
            st.write(match['text'])


# ============================================
# STEP 4: COMPATIBILITY ANALYSIS
# ============================================

st.header("4️⃣ Compatibility Analysis")

if st.button("📊 Analyze Compatibility", type="primary", use_container_width=True):
    if not st.session_state.matches:
        st.error("❌ Please perform matching first (Step 3)")
    else:
        with st.spinner("Analyzing resume-JD compatibility..."):
            try:
                analyzer = MatchingAnalyzer()
                analysis = analyzer.analyze(
                    st.session_state.jd_data,
                    st.session_state.matches,
                    st.session_state.resume_data
                )
                
                # Add enhanced skill match info if available
                if ENHANCED_AVAILABLE and '_skill_match' in st.session_state.resume_data:
                    skill_match = st.session_state.resume_data['_skill_match']
                    analysis['enhanced_skill_match'] = {
                        'percentage': skill_match['match_percentage'],
                        'matched_count': skill_match['matched_count'],
                        'total_jd_skills': skill_match['total_jd_skills'],
                        'total_resume_skills': skill_match['total_resume_skills'],
                        'matched_skills': skill_match['matched_skills'],
                        'missing_skills': skill_match['missing_skills']
                    }
                
                st.session_state.analysis = analysis
                st.success("✓ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Display analysis
if st.session_state.analysis:
    analysis = st.session_state.analysis
    
    # Overall score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{analysis['overall_score']}/100",
            delta=analysis['compatibility_level']
        )
    
    with col2:
        st.metric(
            "Skill Match",
            f"{analysis['skill_match']['percentage']:.0f}%",
            delta=f"{len(analysis['skill_match']['matched'])} skills"
        )
    
    with col3:
        st.metric(
            "Experience Relevance",
            f"{analysis['experience_relevance']['score']:.0f}/100",
            delta=f"{analysis['experience_relevance']['strong_matches']} strong"
        )
    
    # Enhanced skill match info
    if 'enhanced_skill_match' in analysis:
        st.info(f"✨ Enhanced: Extracted {analysis['enhanced_skill_match']['total_resume_skills']} skills | "
                f"Matched {analysis['enhanced_skill_match']['matched_count']}/{analysis['enhanced_skill_match']['total_jd_skills']} "
                f"({analysis['enhanced_skill_match']['percentage']:.0f}%)")
    
    # Detailed breakdown
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("💪 Top Strengths")
        for strength in analysis['top_strengths']:
            st.success(f"✓ {strength}")
        
        # Show enhanced matched skills
        if 'enhanced_skill_match' in analysis and analysis['enhanced_skill_match']['matched_skills']:
            with st.expander(f"✅ Matched Skills ({len(analysis['enhanced_skill_match']['matched_skills'])})"):
                matched_text = ", ".join(analysis['enhanced_skill_match']['matched_skills'])
                st.write(matched_text)
    
    with col_right:
        st.subheader("📚 Skill Gaps")
        
        # Show enhanced missing skills if available
        if 'enhanced_skill_match' in analysis and analysis['enhanced_skill_match']['missing_skills']:
            for skill in analysis['enhanced_skill_match']['missing_skills'][:5]:
                st.error(f"✗ {skill}")
            if len(analysis['enhanced_skill_match']['missing_skills']) > 5:
                st.caption(f"... and {len(analysis['enhanced_skill_match']['missing_skills']) - 5} more")
        elif analysis['gaps']:
            for gap in analysis['gaps']:
                st.error(f"✗ {gap}")
        else:
            st.success("✓ No major gaps!")
    
    # Insights
    st.subheader("💡 Key Insights")
    for insight in analysis['insights']:
        st.info(insight)
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    for i, rec in enumerate(analysis['recommendations'], 1):
        st.markdown(f"{i}. {rec}")
    
    # Download analysis
    analysis_json = json.dumps(analysis, indent=2)
    st.download_button(
        "📥 Download Analysis Report (JSON)",
        data=analysis_json,
        file_name="matching_analysis.json",
        mime="application/json"
    )


# ============================================
# STEP 5: COVER LETTER GENERATION
# ============================================

st.header("5️⃣ Cover Letter Generation")

if st.button("✍️ Generate Cover Letter", type="primary", use_container_width=True):
    if not st.session_state.matches:
        st.error("❌ Please perform matching first (Step 3)")
    elif not config['hf_token'] and config['generation_mode'] != 'rule':
        st.error("❌ Please enter HuggingFace token for LLM generation")
    else:
        with st.spinner("Generating personalized cover letter..."):
            try:
                writer = CoverLetterWriter(
                    model=config['text_model'],
                    api_key=config['hf_token']
                )
                
                # Enhance JD data with analysis insights if available
                jd_data_enhanced = st.session_state.jd_data.copy()
                if st.session_state.analysis:
                    jd_data_enhanced['analysis_insights'] = {
                        'score': st.session_state.analysis['overall_score'],
                        'compatibility': st.session_state.analysis['compatibility_level'],
                        'top_strengths': st.session_state.analysis['top_strengths'],
                        'key_insights': st.session_state.analysis['insights'][:2]
                    }
                
                result = writer.generate_cover_letter(
                    jd_data_enhanced,
                    st.session_state.matches
                )
                
                if result['success']:
                    st.session_state.cover_letter = result['cover_letter']
                    st.success(f"✓ Cover letter generated! ({result['word_count']} words)")
                else:
                    st.session_state.cover_letter = result['cover_letter']
                    st.warning(f"⚠️ {result.get('note', 'Generated using fallback method')}")
                
            except Exception as e:
                st.error(f"❌ Error generating cover letter: {str(e)}")

# Display cover letter
if st.session_state.cover_letter:
    st.subheader("📝 Generated Cover Letter")
    
    # Editable text area
    edited_letter = st.text_area(
        "Edit your cover letter",
        value=st.session_state.cover_letter,
        height=400,
        help="Feel free to customize the generated letter"
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download as TXT",
            data=edited_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Add analysis summary if available
        full_content = edited_letter
        if st.session_state.analysis:
            full_content += f"\n\n{'='*60}\n"
            full_content += "COMPATIBILITY ANALYSIS SUMMARY\n"
            full_content += f"{'='*60}\n"
            full_content += f"Overall Match: {st.session_state.analysis['overall_score']}/100 ({st.session_state.analysis['compatibility_level']})\n"
            full_content += f"Skill Coverage: {st.session_state.analysis['skill_match']['percentage']:.0f}%\n\n"
            full_content += "Top Strengths:\n"
            for s in st.session_state.analysis['top_strengths']:
                full_content += f"  • {s}\n"
        
        st.download_button(
            "📥 Download with Analysis",
            data=full_content,
            file_name="cover_letter_with_analysis.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.caption("""
💡 **Tips**: 
- Use the analysis report to understand your fit before applying
- Customize the generated cover letter with company-specific details
- Highlight skills from your "Top Strengths" in your application
- Enhanced Matcher extracts skills from both resume sections and experiences
""")

footer_text = "Built with ❤️ using JobSpanBERT, Snowflake Arctic Embed, and Zephyr/Mistral"
if ENHANCED_AVAILABLE:
    footer_text += " | Enhanced Matcher ✨"
st.caption(footer_text)