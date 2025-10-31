"""
JD Extractor for International Job Postings
Uses JobSpanBERT for accurate skill extraction from English job descriptions
"""

import re
import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# English stopwords for JD parsing
STOPWORDS = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through"
}

# Key section patterns for English JDs
KEY_PATTERNS = [
    # Requirements/Qualifications section
    r"(?:Requirements?|Qualifications?|What [Ww]e'?re? [Ll]ooking [Ff]or|Skills? [Rr]equired?)[\s\S]*?(?=Responsibilities|Benefits|About|Nice to Have|Preferred|$)",
    
    # Responsibilities section  
    r"(?:Responsibilities|What [Yy]ou'?ll [Dd]o|Your [Rr]ole|Job [Dd]escription)[\s\S]*?(?=Requirements|Benefits|About|Qualifications|$)",
    
    # Nice to have / Preferred
    r"(?:Nice to [Hh]ave|Preferred|Bonus|Plus|Optional)[\s\S]*?(?=Benefits|About|$)"
]

# Company domains to clean from scraped content
NOISE_PATTERNS = [
    r"Apply\s+(?:now|here|today)",
    r"(?:Follow|Like|Share)\s+us\s+on",
    r"Cookie\s+(?:policy|settings?)",
    r"Privacy\s+(?:policy|statement)",
    r"©\s*\d{4}",
    r"All\s+rights\s+reserved"
]


class SkillExtractor:
    """Extract skills using JobSpanBERT NER model"""
    
    def __init__(self, model_name: str = "jjzha/jobspanbert-base-cased"):
        self.model_name = model_name
        self._ner_pipeline = None
        
    @property
    def ner_pipeline(self):
        """Lazy load the NER pipeline"""
        if self._ner_pipeline is None:
            print(f"Loading skill extraction model: {self.model_name}...")
            try:
                self._ner_pipeline = pipeline(
                    "ner",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    aggregation_strategy="simple",
                    device=-1  # Use CPU, change to 0 for GPU
                )
            except Exception as e:
                print(f"Warning: Could not load JobSpanBERT: {e}")
                print("Falling back to keyword extraction")
                self._ner_pipeline = None
        return self._ner_pipeline
    
    def extract_skills(self, text: str, max_length: int = 5000) -> List[str]:
        """
        Extract skills using NER model
        
        Args:
            text: Job description text
            max_length: Maximum text length (for long JDs)
            
        Returns:
            List of unique skills
        """
        # Truncate if too long (JobSpanBERT has token limit)
        if len(text) > max_length:
            text = text[:max_length]
        
        if self.ner_pipeline is None:
            # Fallback to simple keyword extraction
            return self._fallback_extraction(text)
        
        try:
            # Run NER
            entities = self.ner_pipeline(text)
            
            # Extract skill entities
            skills = []
            for entity in entities:
                # Filter by entity type and confidence
                if entity.get('entity_group') in ['Skill', 'SKILL'] and entity.get('score', 0) > 0.5:
                    skill = entity['word'].strip()
                    # Clean up tokenization artifacts
                    skill = skill.replace(' ##', '').replace('##', '')
                    if len(skill) > 1 and skill not in STOPWORDS:
                        skills.append(skill)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_skills = []
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower not in seen:
                    seen.add(skill_lower)
                    unique_skills.append(skill)
            
            return unique_skills[:50]  # Limit to top 50
            
        except Exception as e:
            print(f"Error in skill extraction: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> List[str]:
        """Enhanced skill extraction - works across all industries"""
        skills = set()
        
        # 1. Extract acronyms (PSM, PHA, NFPA, IEC, OSHA, etc.)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        for acronym in acronyms:
            # Filter out common words that are all caps and noise
            if acronym not in {'AND', 'OR', 'THE', 'FOR', 'WITH', 'FROM', 'NOT', 'HAS', 'ARE', 'CAN', 'ALL', 'BUT', 'HIS', 'HER', 'OUR'}:
                skills.add(acronym)
        
        # 2. Extract phrases in parentheses (often skill definitions)
        # e.g., "Process Safety Management (PSM)" → extract both
        paren_matches = re.findall(r'([A-Z][A-Za-z\s&-]+)\s*\(([A-Z]{2,6})\)', text)
        for full_name, abbrev in paren_matches:
            full_name = full_name.strip()
            if len(full_name) > 3:
                skills.add(full_name)
                skills.add(abbrev)
        
        # 3. IT & Software (original patterns)
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|Ruby|Go|Rust|C\+\+|C#|PHP|R|MATLAB|Scala|Kotlin|Swift)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Django|Flask|Spring|\.NET|Express|Laravel)\b',
            r'\b(?:SQL|PostgreSQL|MySQL|MongoDB|Redis|Cassandra|DynamoDB|Oracle|NoSQL)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|CI/CD|Terraform|Ansible)\b',
            r'\b(?:Machine Learning|Deep Learning|NLP|Computer Vision|MLOps|AI)\b',
            r'\b(?:Agile|Scrum|Kanban|JIRA|Confluence|DevOps)\b'
        ]
        
        # 4. Engineering & Technical domains
        engineering_patterns = [
            r'\b(?:Chemical Engineering|Mechanical Engineering|Electrical Engineering|Civil Engineering|Industrial Engineering)\b',
            r'\b(?:Process Safety|Risk Assessment|Risk Management|Safety Management|Hazard Analysis)\b',
            r'\b(?:Project Management|PMP|Six Sigma|Lean|Quality Assurance|QA|Quality Control)\b',
            r'\b(?:Regulatory Compliance|Compliance|Auditing|Standards|Certification)\b',
            r'\b(?:Data Analysis|Statistical Analysis|Analytics|Business Intelligence|BI)\b',
            r'\b(?:CAD|AutoCAD|SolidWorks|MATLAB|Simulink)\b'
        ]
        
        # 5. Business & Management
        business_patterns = [
            r'\b(?:Leadership|Team Management|People Management|Stakeholder Management)\b',
            r'\b(?:Strategic Planning|Business Development|Market Research|Strategy)\b',
            r'\b(?:Communication|Presentation|Negotiation|Problem Solving)\b',
            r'\b(?:Financial Analysis|Budget Management|Cost Control|Budgeting)\b'
        ]
        
        # Apply all patterns
        all_patterns = tech_patterns + engineering_patterns + business_patterns
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update(matches)
        
        # 6. Extract capitalized phrases (likely skills/tools/methods)
        # e.g., "Process Hazard Analyses", "National Fire Protection Association"
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b', text)
        for phrase in cap_phrases:
            phrase = phrase.strip()
            words = phrase.split()
            
            # Filter out common non-skill phrases
            if (len(words) >= 2 and 
                phrase not in {
                    'Bachelor Degree', 'Working Experience', 'Job Requirements',
                    'Nice Have', 'Key Client', 'Minimum Years', 'Job Description',
                    'Job Requirements', 'Technical Knowledge', 'Skills Required',
                    'North America', 'South America', 'United States', 'United Kingdom'
                } and
                # Must not start with generic words
                words[0].lower() not in {'degree', 'job', 'working', 'technical', 'skills', 'minimum', 'years'}):
                skills.add(phrase)
        
        # Clean up and filter
        cleaned_skills = []
        
        # Noise patterns to exclude
        noise_patterns = [
            r'^(and|or|the|with|from|for|about)$',
            r'^\w{1,2}$',  # Single/two letter words (except acronyms already added)
            r'\n',  # Contains newlines
            r'^\d+',  # Starts with number
            r'^(local|cross|standards?)$',  # Too generic
            r'^Job\s',  # Job-related headers
            r'^Working\s',  # Working-related headers
            r'Qualifications?\s*$',  # Section headers
            r'^(in|on|at|to|by)\s'  # Prepositions
        ]
        
        for skill in skills:
            skill = skill.strip()
            
            # Skip if matches noise patterns
            is_noise = False
            for noise_pattern in noise_patterns:
                if re.match(noise_pattern, skill, re.IGNORECASE):
                    is_noise = True
                    break
            
            if is_noise:
                continue
            
            # Remove if too short, too long, or is a stopword
            if 2 <= len(skill) <= 60 and skill.lower() not in STOPWORDS:
                # Remove trailing punctuation
                skill = re.sub(r'[.,;:]+$', '', skill)
                # Must start with alphanumeric
                if skill and (skill[0].isalnum() or skill[0] == '.'):
                    cleaned_skills.append(skill)
        
        # Remove duplicates (case-insensitive)
        seen = set()
        unique_skills = []
        for skill in cleaned_skills:
            skill_lower = skill.lower()
            # Also check if it's a substring of an existing skill
            is_duplicate = False
            for existing in seen:
                if (skill_lower == existing or 
                    (len(skill_lower) > 3 and skill_lower in existing) or
                    (len(existing) > 3 and existing in skill_lower)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills[:50]  # Return top 50


def _extract_sections(text: str) -> List[str]:
    """Extract key sections from JD text"""
    sections = []
    for pattern in KEY_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            section = match.group(0).strip()
            sections.append(section)
    
    # If no sections found, use full text
    return sections if sections else [text]


def _clean_scraped_text(text: str) -> str:
    """Remove common noise from scraped job postings"""
    # Remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_jd_from_url(
    url: str, 
    skill_extractor: Optional[SkillExtractor] = None
) -> Dict:
    """
    Extract job description from URL
    
    Args:
        url: Job posting URL (LinkedIn, Indeed, company sites)
        skill_extractor: Optional SkillExtractor instance (reuse model)
        
    Returns:
        Dict with clean_text, sections, keywords, skills, source_url
    """
    try:
        # Fetch with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Extract main content
        main = soup.find('main') or soup.find('article') or soup.body or soup
        clean_text = main.get_text(separator='\n', strip=True)
        
        # Clean up
        clean_text = _clean_scraped_text(clean_text)
        
        # Extract sections
        sections = _extract_sections(clean_text)
        
        # Extract skills with NER model
        if skill_extractor is None:
            skill_extractor = SkillExtractor()
        
        combined_text = '\n'.join(sections)
        skills = skill_extractor.extract_skills(combined_text)
        
        # Also extract general keywords for backup
        keywords = _extract_keywords(combined_text, exclude_skills=skills)
        
        return {
            "clean_text": clean_text,
            "sections": sections,
            "skills": skills,  # NER-extracted skills
            "keywords": keywords,  # General keywords
            "source_url": url,
            "success": True
        }
        
    except Exception as e:
        print(f"Error extracting from URL: {e}")
        return {
            "clean_text": "",
            "sections": [],
            "skills": [],
            "keywords": [],
            "source_url": url,
            "success": False,
            "error": str(e)
        }


def extract_jd_from_text(
    text: str,
    skill_extractor: Optional[SkillExtractor] = None
) -> Dict:
    """
    Extract job description from plain text
    
    Args:
        text: Job description text
        skill_extractor: Optional SkillExtractor instance
        
    Returns:
        Dict with clean_text, sections, keywords, skills
    """
    # Clean text
    clean_text = _clean_scraped_text(text)
    
    # Extract sections
    sections = _extract_sections(clean_text)
    
    # Extract skills
    if skill_extractor is None:
        skill_extractor = SkillExtractor()
    
    combined_text = '\n'.join(sections)
    skills = skill_extractor.extract_skills(combined_text)
    
    # General keywords
    keywords = _extract_keywords(combined_text, exclude_skills=skills)
    
    return {
        "clean_text": clean_text,
        "sections": sections,
        "skills": skills,
        "keywords": keywords,
        "source": "manual_input",
        "success": True
    }


def _extract_keywords(text: str, exclude_skills: List[str] = None, limit: int = 30) -> List[str]:
    """
    Extract general keywords (non-skill terms)
    Used as backup when skill extraction fails
    """
    if exclude_skills is None:
        exclude_skills = []
    
    # Normalize
    text = text.lower()
    
    # Extract alphanumeric tokens
    tokens = re.findall(r'\b[a-z][a-z0-9+#.-]{2,}\b', text)
    
    # Count frequency
    freq = {}
    for token in tokens:
        if token not in STOPWORDS and token not in [s.lower() for s in exclude_skills]:
            freq[token] = freq.get(token, 0) + 1
    
    # Sort by frequency
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in sorted_keywords[:limit]]


# Convenience function for batch processing
def extract_multiple_jds(
    sources: List[str],
    is_url: bool = True
) -> List[Dict]:
    """
    Extract multiple JDs efficiently (reuses model)
    
    Args:
        sources: List of URLs or text strings
        is_url: Whether sources are URLs (True) or text (False)
        
    Returns:
        List of extraction results
    """
    # Initialize model once
    skill_extractor = SkillExtractor()
    
    results = []
    for source in sources:
        if is_url:
            result = extract_jd_from_url(source, skill_extractor)
        else:
            result = extract_jd_from_text(source, skill_extractor)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Test with example
    sample_jd = """
    Senior Data Engineer - Remote
    
    Requirements:
    - 5+ years of experience with Python and SQL
    - Strong knowledge of Apache Spark and Airflow
    - Experience with AWS (S3, Redshift, Lambda)
    - Excellent communication skills
    
    Responsibilities:
    - Build scalable ETL pipelines
    - Optimize data warehouse performance
    - Collaborate with data scientists
    """
    
    result = extract_jd_from_text(sample_jd)
    print("Extracted Skills:", result['skills'])
    print("Keywords:", result['keywords'])