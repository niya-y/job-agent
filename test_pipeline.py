"""
End-to-end test of the job agent pipeline
Tests JD extraction, matching, analysis, and cover letter generation
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.jd_extractor import extract_jd_from_text, SkillExtractor
from agents.matcher import ResumeMatcher
from agents.analyzer import MatchingAnalyzer
from agents.writer import CoverLetterWriter


# Sample job description for testing
SAMPLE_JD = """
Senior Data Engineer - Remote

About the Role:
We are looking for an experienced Data Engineer to join our growing team. You will be responsible for building and maintaining our data infrastructure.

Requirements:
- 5+ years of experience in data engineering
- Strong proficiency in Python and SQL
- Experience with Apache Spark and distributed computing
- Knowledge of AWS data services (S3, Redshift, Glue)
- Experience with workflow orchestration tools like Airflow
- Strong understanding of data modeling and ETL best practices

Responsibilities:
- Design and implement scalable data pipelines
- Optimize existing ETL workflows for performance
- Collaborate with data scientists and analysts
- Ensure data quality and reliability
- Mentor junior team members

Nice to Have:
- Experience with Kafka or other streaming technologies
- Knowledge of containerization (Docker, Kubernetes)
- Familiarity with data visualization tools
- Background in machine learning operations (MLOps)
"""


def test_jd_extraction():
    """Test 1: JD Extraction with JobSpanBERT"""
    print("\n" + "="*60)
    print("TEST 1: JD EXTRACTION")
    print("="*60)
    
    # Extract JD
    print("\n📄 Extracting job description...")
    jd_data = extract_jd_from_text(SAMPLE_JD)
    
    print(f"\n✓ Extraction successful: {jd_data['success']}")
    print(f"\n📊 Extracted {len(jd_data['skills'])} skills:")
    for i, skill in enumerate(jd_data['skills'][:10], 1):
        print(f"  {i}. {skill}")
    
    if len(jd_data['skills']) > 10:
        print(f"  ... and {len(jd_data['skills']) - 10} more")
    
    print(f"\n📑 Found {len(jd_data['sections'])} sections")
    
    return jd_data


def test_resume_matching(jd_data):
    """Test 2: Resume Matching with Snowflake Arctic Embed"""
    print("\n" + "="*60)
    print("TEST 2: RESUME MATCHING")
    print("="*60)
    
    # Load resume
    print("\n📋 Loading resume...")
    resume_path = "data/resume.json"
    
    if not os.path.exists(resume_path):
        print(f"❌ Resume not found at {resume_path}")
        print("Please create data/resume.json following the format in README")
        return None, None
    
    with open(resume_path) as f:
        resume = json.load(f)
    
    print(f"✓ Loaded resume with {len(resume.get('experiences', []))} experiences")
    
    # Initialize matcher
    print("\n🔍 Initializing matcher...")
    api_key = os.getenv("HF_TOKEN")
    
    if not api_key:
        print("❌ HF_TOKEN not found in environment")
        print("Please set HF_TOKEN in .env file")
        return None, None
    
    matcher = ResumeMatcher(
        embed_model="Snowflake/snowflake-arctic-embed-m",
        api_key=api_key
    )
    
    # Build index
    print("\n🗂️ Building FAISS index...")
    matcher.build_index_from_resume(resume)
    
    # Search for matches
    print("\n🎯 Finding relevant experiences...")
    matches = matcher.search_by_skills(jd_data['skills'], top_k=6)
    
    print(f"\n✓ Found {len(matches)} matching experiences:")
    for i, match in enumerate(matches, 1):
        score = match['score']
        text = match['text'][:100] + "..." if len(match['text']) > 100 else match['text']
        print(f"\n  {i}. Score: {score:.3f}")
        print(f"     {text}")
    
    return matches, resume


def test_matching_analysis(jd_data, matches, resume):
    """Test 2.5: Matching Analysis - NEW!"""
    print("\n" + "="*60)
    print("TEST 2.5: MATCHING ANALYSIS")
    print("="*60)
    
    if not matches:
        print("⚠️ No matches available, skipping analysis")
        return None
    
    # Initialize analyzer
    print("\n📊 Initializing analyzer...")
    analyzer = MatchingAnalyzer()
    
    # Analyze compatibility
    print("\n🔬 Analyzing resume-JD compatibility...")
    analysis = analyzer.analyze(jd_data, matches, resume)
    
    # Print detailed report
    print("\n" + analyzer.generate_text_report(analysis))
    
    # Save analysis to file
    output_path = "matching_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Full analysis saved to {output_path}")
    
    return analysis


def test_cover_letter_generation(jd_data, matches, analysis=None):
    """Test 3: Cover Letter Generation with LLM"""
    print("\n" + "="*60)
    print("TEST 3: COVER LETTER GENERATION")
    print("="*60)
    
    if not matches:
        print("⚠️ No matches available, skipping cover letter generation")
        return
    
    # Initialize writer
    print("\n✏️ Initializing LLM writer...")
    api_key = os.getenv("HF_TOKEN")
    
    # Choose model (with fallback options)
    models_to_try = [
        "HuggingFaceH4/zephyr-7b-beta",  # Primary choice       
        "meta-llama/Meta-Llama-3.1-8B-Instruct" # Fallback 1
        "mistralai/Mistral-7B-Instruct-v0.3"# Fallback 2 (may have access issues)
    ]
    
    writer = None
    for model in models_to_try:
        try:
            print(f"   Trying {model}...")
            writer = CoverLetterWriter(model=model, api_key=api_key)
            print(f"   ✓ Successfully initialized with {model}")
            break
        except Exception as e:
            print(f"   ⚠️ {model} not available: {e}")
            continue
    
    if not writer:
        print("❌ No available LLM models found")
        print("Falling back to rule-based generation...")
        writer = CoverLetterWriter(api_key=api_key)
    
    # Prepare context with analysis insights
    enhanced_jd_data = jd_data.copy()
    if analysis:
        enhanced_jd_data['analysis_insights'] = {
            'score': analysis['overall_score'],
            'compatibility': analysis['compatibility_level'],
            'top_strengths': analysis['top_strengths'],
            'key_insights': analysis['insights'][:2]  # Top 2 insights
        }
    
    # Generate cover letter
    print("\n📝 Generating cover letter (this may take 15-30 seconds)...")
    
    try:
        result = writer.generate_cover_letter(enhanced_jd_data, matches)
        
        if result['success']:
            print("\n✓ Cover letter generated successfully!")
            print(f"   Word count: {result['word_count']}")
            print(f"   Model: {result['model_used']}")
            
            print("\n" + "-"*60)
            print("GENERATED COVER LETTER:")
            print("-"*60)
            print(result['cover_letter'])
            print("-"*60)
            
            # Save to file
            output_path = "generated_cover_letter.txt"
            with open(output_path, "w") as f:
                f.write(result['cover_letter'])
                
                # Append analysis summary
                if analysis:
                    f.write("\n\n" + "="*60)
                    f.write("\n📊 COMPATIBILITY ANALYSIS SUMMARY\n")
                    f.write("="*60 + "\n")
                    f.write(f"Overall Match: {analysis['overall_score']}/100 ({analysis['compatibility_level']})\n")
                    f.write(f"Skill Coverage: {analysis['skill_match']['percentage']:.0f}%\n\n")
                    f.write("Top Strengths:\n")
                    for strength in analysis['top_strengths']:
                        f.write(f"  • {strength}\n")
            
            print(f"\n💾 Saved to {output_path}")
        else:
            print(f"\n⚠️ Generation used fallback: {result.get('note', 'Unknown')}")
            print("\nGenerated text:")
            print(result['cover_letter'])
    
    except Exception as e:
        print(f"\n❌ Error generating cover letter: {e}")
        print("\nThis is likely due to:")
        print("  1. Invalid HuggingFace token")
        print("  2. Model not available in Inference API")
        print("  3. Rate limiting")
        print("\nYou can still use the extracted skills and matches manually!")


def main():
    """Run full pipeline test"""
    print("\n" + "="*60)
    print("🚀 JOB AGENT PIPELINE TEST")
    print("="*60)
    
    # Check for HF token
    if not os.getenv("HF_TOKEN"):
        print("\n❌ Error: HF_TOKEN not found")
        print("\nPlease:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your HuggingFace token")
        print("  3. Get token from: https://huggingface.co/settings/tokens")
        return
    
    try:
        # Test 1: JD Extraction
        jd_data = test_jd_extraction()
        
        # Test 2: Resume Matching
        matches, resume = test_resume_matching(jd_data)
        
        # Test 2.5: Matching Analysis (NEW!)
        analysis = None
        if matches:
            analysis = test_matching_analysis(jd_data, matches, resume)
        
        # Test 3: Cover Letter Generation (with analysis insights)
        if matches:
            test_cover_letter_generation(jd_data, matches, analysis)
        
        print("\n" + "="*60)
        print("✅ PIPELINE TEST COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        print("  1. generated_cover_letter.txt - Cover letter with analysis summary")
        print("  2. matching_analysis.json - Detailed compatibility report")
        print("\nNext steps:")
        print("  1. Review the matching analysis to understand your fit")
        print("  2. Review the generated cover letter")
        print("  3. Customize with your personal details")
        print("  4. Try with your own job postings!")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()