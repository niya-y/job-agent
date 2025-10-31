"""
End-to-end test of the job agent pipeline
Tests JD extraction, matching, analysis, and cover letter generation
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    print("⚠️  enhanced_matcher not found. Using basic skill extraction.")
    ENHANCED_AVAILABLE = False


# Sample job description for testing
SAMPLE_JD = """
Job Description
Role Summary :
This role ensures the stable and efficient operation of local IT infrastructure while actively executing corporate and regional IT rollout plans. Reporting to the Local IS Manager, the Infrastructure Manager supports daily operations (L1) and coordinates with global/regional teams for escalations (L2/L3). This position serves as the local owner of rollout activities, overseeing implementation, local adaptation, testing, and end-user readiness to ensure successful deployment aligned with corporate IT strategy.
 
KEY RESPONSIBILITIES :
- Manage and monitor daily IT infrastructure operations (network, servers, store devices, Wi-Fi, etc.)
- Provide L1 support and escalate unresolved issues to L2/L3 teams or Local IS Manager
- Lead local execution of global/regional infrastructure rollout plans
- Coordinate rollout schedules, testing, and user communication to ensure business continuity
- Collaborate with corporate IT teams to localize global initiatives and ensure compliance
- Oversee IT hardware/software procurement and maintain accurate inventory
- Liaise with external vendors for support, maintenance, and installations
- Assist in onboarding and supporting new systems, infrastructure projects, and upgrades
- Manage documentation related to local infrastructure configurations and rollouts
- Support IT expense reporting, purchase processing, and vendor billing coordination


Job Requirements
- Bachelor’s degree in IT or related field
- 3–5 years of experience in IT infrastructure or technical operations
- Familiarity with global rollout plans and ability to localize for business environments
- Knowledge of networking, endpoint management, server support, and IT asset handling
- Effective communicator with experience in global team coordination
- Strong organizational and documentation skills
- Fluent in Korean; English required for global IT alignment and reporting
- Experience in retail IT infrastructure is preferred

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
    """Test 2: Resume Matching with Enhanced Skill Extraction + Snowflake Arctic Embed"""
    print("\n" + "="*60)
    print("TEST 2: RESUME MATCHING (ENHANCED)")
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
    
    # ============================================================
    # STEP 1: Enhanced Skill Extraction (NEW!)
    # ============================================================
    if ENHANCED_AVAILABLE:
        print("\n🔍 Extracting ALL skills from resume (Enhanced Matcher)...")
        
        # Extract skills from both 'skills' section and 'experiences'
        all_skills = extract_resume_skills(resume)
        explicit_skills = resume.get('skills', [])
        
        print(f"✓ Skill extraction complete:")
        print(f"   • From skills section: {len(explicit_skills)} skills")
        print(f"   • Total extracted: {len(all_skills)} skills")
        print(f"   • Improvement: {len(all_skills) / max(len(explicit_skills), 1):.1f}x more skills!")
        
        # Show some extracted skills
        if len(all_skills) > 10:
            print(f"\n   Sample skills: {', '.join(all_skills[:10])}...")
        else:
            print(f"\n   Skills: {', '.join(all_skills)}")
        
        # Get direct skill match score
        print("\n📊 Calculating skill match with JD...")
        skill_match = match_resume_to_jd_skills(resume, jd_data['skills'])
        print(f"✓ Skill match: {skill_match['match_percentage']:.1f}%")
        print(f"   • Matched: {skill_match['matched_count']}/{skill_match['total_jd_skills']} skills")
        
        if skill_match['matched_skills'][:5]:
            print(f"   • Top matches: {', '.join(skill_match['matched_skills'][:5])}")
        
        # Use extracted skills for semantic search
        search_skills = all_skills
    else:
        print("\n⚠️  Enhanced matcher not available. Using skills section only.")
        search_skills = resume.get('skills', [])
        skill_match = None
    
    # ============================================================
    # STEP 2: Semantic Search with Snowflake Arctic Embed
    # ============================================================
    print("\n🔍 Initializing Snowflake Arctic Embed matcher...")
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
    print("\n🗂️  Building FAISS index from resume experiences...")
    matcher.build_index_from_resume(resume)
    
    # Search for matches using ALL extracted skills
    # Ensure search_skills is a list (not set or other type)
    if not isinstance(search_skills, list):
        search_skills = list(search_skills)
    
    print(f"\n🎯 Finding relevant experiences with {len(search_skills)} skills...")
    matches = matcher.search_by_skills(search_skills, top_k=6)
    
    print(f"\n✓ Found {len(matches)} matching experiences:")
    for i, match in enumerate(matches, 1):
        score = match['score']
        text = match['text'][:100] + "..." if len(match['text']) > 100 else match['text']
        print(f"\n  {i}. Score: {score:.3f}")
        print(f"     {text}")
    
    # Store skill match info for later use
    if ENHANCED_AVAILABLE and skill_match:
        resume['_skill_match'] = skill_match
    
    return matches, resume


def test_matching_analysis(jd_data, matches, resume):
    """Test 2.5: Matching Analysis - Enhanced with Skill Match Info"""
    print("\n" + "="*60)
    print("TEST 2.5: MATCHING ANALYSIS")
    print("="*60)
    
    if not matches:
        print("⚠️  No matches available, skipping analysis")
        return None
    
    # Initialize analyzer
    print("\n📊 Initializing analyzer...")
    analyzer = MatchingAnalyzer()
    
    # Analyze compatibility
    print("\n🔬 Analyzing resume-JD compatibility...")
    analysis = analyzer.analyze(jd_data, matches, resume)
    
    # Add enhanced skill match info if available
    if ENHANCED_AVAILABLE and '_skill_match' in resume:
        skill_match = resume['_skill_match']
        print("\n📈 Enhanced Skill Analysis:")
        print(f"   • Direct skill match: {skill_match['match_percentage']:.1f}%")
        print(f"   • Matched skills: {skill_match['matched_count']}/{skill_match['total_jd_skills']}")
        print(f"   • Total resume skills: {skill_match['total_resume_skills']}")
        
        # Add to analysis result
        analysis['enhanced_skill_match'] = {
            'percentage': skill_match['match_percentage'],
            'matched_count': skill_match['matched_count'],
            'total_jd_skills': skill_match['total_jd_skills'],
            'total_resume_skills': skill_match['total_resume_skills'],
            'matched_skills': skill_match['matched_skills'],
            'missing_skills': skill_match['missing_skills']
        }
    
    # Print detailed report
    print("\n" + analyzer.generate_text_report(analysis))
    
    # Print enhanced skill match details if available
    if ENHANCED_AVAILABLE and '_skill_match' in resume:
        skill_match = resume['_skill_match']
        print("\n" + "="*60)
        print("DETAILED SKILL MATCHING (ENHANCED)")
        print("="*60)
        
        print(f"\n✅ Matched Skills ({len(skill_match['matched_skills'])}):")
        for skill in skill_match['matched_skills']:
            print(f"   • {skill}")
        
        if skill_match['missing_skills']:
            print(f"\n❌ Missing Skills ({len(skill_match['missing_skills'])}):")
            for skill in skill_match['missing_skills'][:10]:
                print(f"   • {skill}")
            if len(skill_match['missing_skills']) > 10:
                print(f"   ... and {len(skill_match['missing_skills']) - 10} more")
    
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
    
    # Check for enhanced matcher
    if ENHANCED_AVAILABLE:
        print("\n✅ Enhanced Matcher enabled")
        print("   • Extracts skills from experiences section")
        print("   • 3x better matching accuracy")
    else:
        print("\n⚠️  Enhanced Matcher not found")
        print("   • Using basic skill extraction")
        print("   • For better results, add enhanced_matcher.py")
    
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
        
        # Test 2: Resume Matching (Enhanced!)
        matches, resume = test_resume_matching(jd_data)
        
        # Test 2.5: Matching Analysis (with skill match info)
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
        
        if ENHANCED_AVAILABLE:
            print("\n💡 Enhanced Features Used:")
            print("  • Skills extracted from experiences section")
            print("  • Direct skill matching with JD")
            print("  • Combined semantic + keyword matching")
        
        print("\nNext steps:")
        print("  1. Review the matching analysis to understand your fit")
        print("  2. Review the generated cover letter")
        print("  3. Customize with your personal details")
        print("  4. Try with your own job postings!")
        
        if not ENHANCED_AVAILABLE:
            print("\n💡 Tip: Add enhanced_matcher.py for better results!")
            print("   → 3x better skill extraction")
            print("   → More accurate matching")
        
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()