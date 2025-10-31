"""
Enhanced Resume Matcher with Skill Extraction
Extracts skills from both 'skills' section and 'experiences' section
Matches with JD skills using 4-strategy algorithm
"""

import re
from typing import List, Dict, Set, Optional
from collections import Counter


class ResumeSkillExtractor:
    """
    Extracts skills from resume using 4 strategies:
    1. Explicit skills section
    2. Acronym extraction
    3. Skill library matching
    4. Pattern-based extraction
    """
    
    # Comprehensive skill library (200+ keywords)
    SKILL_LIBRARY = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 
        'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'php', 'perl',
        
        # Web Technologies
        'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 
        'spring', 'asp.net', 'html', 'css', 'sass', 'webpack', 'babel',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 
        'dynamodb', 'elasticsearch', 'oracle', 'sql server', 'sqlite',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 
        'terraform', 'ansible', 'ci/cd', 'devops', 'microservices',
        
        # Data Science & ML
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
        'scikit-learn', 'pandas', 'numpy', 'data analysis', 'statistics',
        'nlp', 'computer vision', 'neural networks', 'ai', 'ml', 'dl',
        
        # Big Data
        'spark', 'hadoop', 'kafka', 'airflow', 'flink', 'hive', 'presto',
        'apache spark', 'apache kafka', 'apache airflow', 'etl', 'data pipeline',
        
        # Analytics & BI
        'tableau', 'power bi', 'looker', 'qlik', 'excel', 'data visualization',
        'dashboard', 'reporting', 'bi', 'analytics',
        
        # Testing & Quality
        'junit', 'pytest', 'selenium', 'jest', 'testing', 'unit testing',
        'integration testing', 'qa', 'quality assurance', 'tdd',
        
        # Methodologies
        'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'six sigma',
        
        # Soft Skills (selected technical-adjacent ones)
        'leadership', 'communication', 'problem solving', 'analytical',
        'project management', 'team collaboration',
        
        # Other Technologies
        'git', 'github', 'rest api', 'graphql', 'json', 'xml', 'api',
        'linux', 'unix', 'bash', 'shell scripting', 'powershell',
    }
    
    # Noise words to exclude
    NOISE_ACRONYMS = {
        'THE', 'AND', 'FOR', 'KEY', 'JOB', 'NEW', 'OLD', 'BIG', 'TOP',
        'ALL', 'ANY', 'OUR', 'YOU', 'WAS', 'ARE', 'CAN', 'BUT', 'NOT',
        'INC', 'LLC', 'LTD', 'USA', 'PDF', 'DOC', 'PPT', 'XLS'
    }
    
    EXCLUDED_PHRASES = [
        'the company', 'the team', 'my role', 'my responsibilities',
        'job duties', 'team members', 'this role', 'this position'
    ]
    
    def __init__(self):
        """Initialize the extractor"""
        # Compile regex patterns for efficiency
        self.acronym_pattern = re.compile(r'\b([A-Z]{2,})\b')
        
        # Pattern for "using X", "with Y", "experience in Z"
        self.skill_patterns = [
            re.compile(r'(?:using|with|in)\s+([A-Z][A-Za-z\s\-\.#\+]{2,20})', re.IGNORECASE),
            re.compile(r'(?:developed|built|created|implemented|designed)\s+([A-Z][A-Za-z\s\-\.#\+]{2,20})\s+(?:using|with)', re.IGNORECASE),
            re.compile(r'(?:experience|proficiency|expertise)\s+(?:in|with)\s+([A-Z][A-Za-z\s\-\.#\+]{2,20})', re.IGNORECASE),
        ]
    
    def extract_all_skills(self, resume_data: Dict) -> Dict:
        """
        Extract skills from entire resume
        
        Args:
            resume_data: Resume JSON with 'skills' and/or 'experiences' fields
            
        Returns:
            {
                'all_skills': [...],           # All unique skills
                'explicit_skills': [...],      # From skills section
                'experience_skills': [...],    # From experiences
                'breakdown': {...}             # Detailed breakdown
            }
        """
        # Strategy 1: Explicit skills section
        explicit_skills = self._extract_explicit_skills(resume_data)
        
        # Strategy 2-4: From experiences
        experience_skills = self._extract_experience_skills(resume_data)
        
        # Combine and deduplicate
        all_skills = self._deduplicate_skills(explicit_skills + experience_skills)
        
        return {
            'all_skills': all_skills,
            'explicit_skills': list(set(explicit_skills)),
            'experience_skills': list(set(experience_skills)),
            'breakdown': {
                'from_skills_section': len(set(explicit_skills)),
                'from_experiences': len(set(experience_skills)),
                'total_unique': len(all_skills)
            }
        }
    
    def _extract_explicit_skills(self, resume_data: Dict) -> List[str]:
        """
        Strategy 1: Extract from explicit 'skills' section
        """
        skills = []
        
        if 'skills' not in resume_data:
            return skills
        
        skill_data = resume_data['skills']
        
        # Handle different formats
        if isinstance(skill_data, list):
            # List format: ["Python", "SQL"]
            skills.extend([str(s).strip() for s in skill_data])
            
        elif isinstance(skill_data, dict):
            # Dict format: {"programming": ["Python"], "tools": ["Docker"]}
            for category, skill_list in skill_data.items():
                if isinstance(skill_list, list):
                    skills.extend([str(s).strip() for s in skill_list])
                    
        elif isinstance(skill_data, str):
            # String format: "Python, SQL, Machine Learning"
            skills.extend([s.strip() for s in skill_data.split(',')])
        
        return [self._normalize_skill(s) for s in skills if s]
    
    def _extract_experience_skills(self, resume_data: Dict) -> List[str]:
        """
        Strategy 2-4: Extract from experiences section
        """
        skills = []
        
        if 'experiences' not in resume_data:
            return skills
        
        experiences = resume_data['experiences']
        if not isinstance(experiences, list):
            return skills
        
        # Combine all experience text
        text_blocks = []
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            
            # Try different field names
            for field in ['description', 'duties', 'responsibilities', 
                         'achievements', 'details', 'summary']:
                if field in exp and exp[field]:
                    text_blocks.append(str(exp[field]))
        
        full_text = ' '.join(text_blocks)
        
        if not full_text:
            return skills
        
        # Strategy 2: Acronym extraction
        skills.extend(self._extract_acronyms(full_text))
        
        # Strategy 3: Skill library matching
        skills.extend(self._match_skill_library(full_text))
        
        # Strategy 4: Pattern-based extraction
        skills.extend(self._extract_pattern_skills(full_text))
        
        return skills
    
    def _extract_acronyms(self, text: str) -> List[str]:
        """
        Strategy 2: Extract acronyms (2+ uppercase letters)
        """
        acronyms = self.acronym_pattern.findall(text)
        
        # Filter noise
        valid_acronyms = [
            acr for acr in acronyms 
            if acr not in self.NOISE_ACRONYMS and len(acr) <= 10
        ]
        
        return valid_acronyms
    
    def _match_skill_library(self, text: str) -> List[str]:
        """
        Strategy 3: Match against skill library
        """
        text_lower = text.lower()
        matched_skills = []
        
        for skill in self.SKILL_LIBRARY:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched_skills.append(skill)
        
        return [self._normalize_skill(s) for s in matched_skills]
    
    def _extract_pattern_skills(self, text: str) -> List[str]:
        """
        Strategy 4: Pattern-based extraction
        Patterns like "using X", "with Y", "experience in Z"
        """
        extracted = []
        
        for pattern in self.skill_patterns:
            matches = pattern.findall(text)
            extracted.extend(matches)
        
        # Clean and filter
        cleaned = []
        for skill in extracted:
            skill = skill.strip()
            
            # Skip if too short or contains excluded phrases
            if len(skill) < 2:
                continue
            
            if any(phrase in skill.lower() for phrase in self.EXCLUDED_PHRASES):
                continue
            
            # Truncate at common stop words
            for stop in [' and ', ' or ', ' to ', ' for ', ' with ']:
                if stop in skill.lower():
                    skill = skill[:skill.lower().index(stop)]
            
            cleaned.append(skill.strip())
        
        return [self._normalize_skill(s) for s in cleaned if s]
    
    def _normalize_skill(self, skill: str) -> str:
        """
        Normalize skill capitalization
        - Acronyms: keep uppercase (SQL, AWS)
        - Others: title case (Machine Learning)
        """
        skill = skill.strip()
        
        # If all uppercase and 2-10 chars, keep as acronym
        if skill.isupper() and 2 <= len(skill) <= 10:
            return skill
        
        # Otherwise use title case
        return skill.title()
    
    def _deduplicate_skills(self, skills: List[str]) -> List[str]:
        """
        Remove duplicates while preserving order
        Case-insensitive comparison
        """
        seen = set()
        unique = []
        
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique.append(skill)
        
        return unique


class EnhancedMatcher:
    """
    Matches resume skills with JD skills
    Uses ResumeSkillExtractor for comprehensive skill extraction
    """
    
    def __init__(self):
        """Initialize the matcher"""
        self.extractor = ResumeSkillExtractor()
    
    def match(
        self, 
        resume_data: Dict, 
        jd_skills: List[str],
        match_threshold: float = 0.0
    ) -> Dict:
        """
        Match resume with JD skills
        
        Args:
            resume_data: Resume JSON
            jd_skills: List of skills from JD
            match_threshold: Minimum match percentage (0.0-1.0)
            
        Returns:
            {
                'match_percentage': 66.7,
                'matched_skills': [...],
                'missing_skills': [...],
                'matched_count': 6,
                'missing_count': 3,
                'total_jd_skills': 9,
                'total_resume_skills': 18,
                'resume_skill_breakdown': {...},
                'all_resume_skills': [...]
            }
        """
        # Extract all skills from resume
        extracted = self.extractor.extract_all_skills(resume_data)
        resume_skills = extracted['all_skills']
        
        # Normalize JD skills
        jd_skills_normalized = [
            self.extractor._normalize_skill(s) for s in jd_skills
        ]
        
        # Find matches (case-insensitive)
        resume_lower = {s.lower(): s for s in resume_skills}
        jd_lower = {s.lower(): s for s in jd_skills_normalized}
        
        matched_keys = set(resume_lower.keys()) & set(jd_lower.keys())
        missing_keys = set(jd_lower.keys()) - set(resume_lower.keys())
        
        matched_skills = sorted([jd_lower[k] for k in matched_keys])
        missing_skills = sorted([jd_lower[k] for k in missing_keys])
        
        # Calculate percentage
        if len(jd_skills_normalized) == 0:
            match_percentage = 0.0
        else:
            match_percentage = (len(matched_skills) / len(jd_skills_normalized)) * 100
        
        return {
            'match_percentage': round(match_percentage, 1),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'matched_count': len(matched_skills),
            'missing_count': len(missing_skills),
            'total_jd_skills': len(jd_skills_normalized),
            'total_resume_skills': len(resume_skills),
            'resume_skill_breakdown': extracted['breakdown'],
            'all_resume_skills': resume_skills
        }
    
    def get_detailed_report(
        self, 
        resume_data: Dict, 
        jd_skills: List[str]
    ) -> str:
        """
        Generate detailed matching report
        
        Returns:
            Formatted string report
        """
        result = self.match(resume_data, jd_skills)
        
        report = f"""
📊 Resume-JD Skill Matching Report
{'=' * 50}

Match Percentage: {result['match_percentage']}%

✅ Matched Skills ({result['matched_count']}):
"""
        for skill in result['matched_skills']:
            report += f"  • {skill}\n"
        
        if result['missing_skills']:
            report += f"\n❌ Missing Skills ({result['missing_count']}):\n"
            for skill in result['missing_skills']:
                report += f"  • {skill}\n"
        
        report += f"""
📋 Resume Skill Breakdown:
  • From skills section: {result['resume_skill_breakdown']['from_skills_section']}
  • From experiences: {result['resume_skill_breakdown']['from_experiences']}
  • Total unique: {result['resume_skill_breakdown']['total_unique']}

📝 All Extracted Skills:
"""
        
        # Show skills in groups of 5
        skills = result['all_resume_skills']
        for i in range(0, len(skills), 5):
            report += "  " + ", ".join(skills[i:i+5]) + "\n"
        
        return report


# Convenience functions
def extract_resume_skills(resume_data: Dict) -> List[str]:
    """
    Quick function to extract all skills from resume
    
    Args:
        resume_data: Resume JSON
        
    Returns:
        List of all extracted skills
    """
    extractor = ResumeSkillExtractor()
    result = extractor.extract_all_skills(resume_data)
    all_skills = result['all_skills']
    
    # Ensure it's a list (not set or other type)
    if not isinstance(all_skills, list):
        all_skills = list(all_skills)
    
    return all_skills


def match_resume_to_jd_skills(
    resume_data: Dict,
    jd_skills: List[str]
) -> Dict:
    """
    Quick matching function
    
    Args:
        resume_data: Resume JSON
        jd_skills: List of JD skills
        
    Returns:
        Matching result dict
    """
    matcher = EnhancedMatcher()
    return matcher.match(resume_data, jd_skills)


if __name__ == "__main__":
    # Test example
    sample_resume = {
        "skills": ["Python", "SQL"],
        "experiences": [
            {
                "company": "Tech Corp",
                "title": "Data Scientist",
                "description": """
                - Developed machine learning models using TensorFlow and PyTorch
                - Built data pipelines with Apache Spark and Airflow
                - Created dashboards using Tableau and Power BI
                - Conducted statistical analysis with Python and R
                """
            }
        ]
    }
    
    sample_jd_skills = [
        "Python", "SQL", "Machine Learning", "TensorFlow",
        "Tableau", "Data Analysis", "Apache Spark", "Statistics", "R"
    ]
    
    print("Testing Enhanced Matcher...")
    print("=" * 50)
    
    # Test skill extraction
    extractor = ResumeSkillExtractor()
    extracted = extractor.extract_all_skills(sample_resume)
    
    print(f"\nExtracted Skills: {len(extracted['all_skills'])}")
    print(f"  - From skills section: {extracted['breakdown']['from_skills_section']}")
    print(f"  - From experiences: {extracted['breakdown']['from_experiences']}")
    print(f"\nAll skills: {', '.join(extracted['all_skills'][:10])}...")
    
    # Test matching
    matcher = EnhancedMatcher()
    report = matcher.get_detailed_report(sample_resume, sample_jd_skills)
    print(report)