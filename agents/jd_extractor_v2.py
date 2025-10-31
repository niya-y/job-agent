"""
Enhanced JD Extractor - Algorithm Overhaul
Simplified, more effective skill extraction using multiple strategies
"""

import re
from typing import Dict, List, Set, Optional
from collections import Counter

# Optional imports
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False


class SkillExtractor:
    """
    New algorithm: Multi-strategy skill extraction
    1. Regex-based pattern matching (fast, reliable)
    2. NER model (accurate, context-aware) - optional
    3. Chunk-based extraction (phrase-level)
    """
    
    def __init__(self, use_model: str = "auto"):
        """
        Args:
            use_model: "jobberta", "nucha", "distilbert", "jobspanbert", or "auto"
        """
        self.use_model = use_model
        self._pipeline = None
        
        # Model selection priority (best to fallback)
        self.model_priority = {
            "jobberta": "jjzha/jobberta-base",  # 2024 EACL - BEST
            "nucha": "Nucha/Nucha_SkillNER_BERT",  # Skill-specific
            "distilbert": "afrodp95/distilbert-base-uncased-finetuned-job-skills-ner",  # Fast
            "jobspanbert": "jjzha/jobspanbert-base-cased",  # Original
        }
    
    def extract_skills(self, text: str, strategy: str = "hybrid") -> List[str]:
        """
        Extract skills using specified strategy
        
        Args:
            text: Job description text
            strategy: "regex", "ner", "hybrid" (default)
        
        Returns:
            List of extracted skills
        """
        if strategy == "regex":
            return self._extract_regex(text)
        elif strategy == "ner" and HAS_TRANSFORMERS:
            return self._extract_ner(text)
        else:  # hybrid
            # Combine both methods for best results
            regex_skills = set(self._extract_regex(text))
            if HAS_TRANSFORMERS:
                ner_skills = set(self._extract_ner(text))
                all_skills = regex_skills.union(ner_skills)
            else:
                all_skills = regex_skills
            
            return list(all_skills)[:50]
    
    def _extract_regex(self, text: str) -> List[str]:
        """
        SIMPLIFIED REGEX EXTRACTION
        Focus on high-confidence patterns only
        """
        skills = set()
        
        # === STRATEGY 1: Extract skill entities directly ===
        
        # 1A. Acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        noise_acronyms = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'ARE', 'CAN', 'NOT',
            'USA', 'CEO', 'CFO', 'CTO', 'VP', 'HR', 'CV', 'PDF', 'DOC',
            'KEY', 'ALL', 'NEW', 'TOP', 'OUR', 'ANY', 'MAIN',
            'HAS', 'WAS', 'ONE', 'TWO', 'WHO', 'HOW', 'WHY', 'WHAT',
            'JOB', 'WORK', 'TEAM', 'ROLE', 'YOUR', 'MUST', 'WILL',
            'MAY', 'SHOULD', 'WOULD', 'COULD'
        }
        skills.update(a for a in acronyms if a not in noise_acronyms)
        
        # 1B. Parenthetical definitions: "Process Safety (PSM)"
        paren_matches = re.findall(
            r'([A-Z][A-Za-z\s&]{3,40})\s*\(([A-Z]{2,6})\)',
            text
        )
        for full_name, abbrev in paren_matches:
            skills.add(full_name.strip())
            skills.add(abbrev)
        
        # === STRATEGY 2: Domain-specific skill libraries ===
        
        # 2A. Programming & Tech
        tech_skills = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust',
            'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Oracle','Streamlit',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git',
            'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring',
            'TensorFlow', 'PyTorch', 'Spark', 'Hadoop', 'Kafka', 'Airflow',
            'REST API', 'GraphQL', 'Microservices', 'CI/CD', 'DevOps','SQLAlchemy'
        ]
        
        # 2B. Engineering & Technical
        engineering_skills = [
            'AutoCAD', 'SolidWorks', 'MATLAB', 'Simulink', 'ANSYS',
            'Six Sigma', 'Lean', 'Kaizen', 'PLC', 'SCADA',
        ]
        
        # 2C. Standards & Compliance
        standards_skills = [
            'ISO', 'GMP', 'cGMP', 'FDA', 'HIPAA', 'GDPR', 'CCPA',
            'SOX', 'OSHA', 'NFPA', 'IEC', 'IEEE',
        ]
        
        # 2D. Project Management
        pm_skills = [
            'PMP', 'Agile', 'Scrum', 'Kanban', 'JIRA', 'Confluence',
            'Waterfall', 'PRINCE2',
        ]
        
        # 2E. Business Tools
        business_skills = [
            'Excel', 'PowerPoint', 'Tableau', 'Power BI', 'Salesforce',
            'SAP', 'Oracle', 'Workday', 'HubSpot',
        ]

        #2F. AI/ML Skills
        ai_ml_skills = [
            # Core AI/ML
            'Machine Learning', 'Deep Learning', 'Artificial Intelligence', 'AI',
            'Neural Networks', 'Reinforcement Learning', 'Transfer Learning',
            'Supervised Learning', 'Unsupervised Learning', 'Semi-supervised Learning',
    
            # Frameworks & Libraries
            'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'scikit-learn',
            'XGBoost', 'LightGBM', 'CatBoost', 'Hugging Face', 'HuggingFace',
            'OpenAI', 'LangChain', 'LlamaIndex',
    
            # NLP
            'Natural Language Processing', 'NLP', 'Text Mining',
            'Sentiment Analysis', 'Named Entity Recognition', 'NER',
            'Topic Modeling', 'Word Embeddings', 'BERT', 'GPT', 'Transformer',
            'Large Language Models', 'LLM', 'Generative AI', 'GenAI',
            'Prompt Engineering', 'RAG', 'Retrieval Augmented Generation',
    
            # Computer Vision
            'Computer Vision', 'Image Processing', 'Object Detection',
            'Image Classification', 'Semantic Segmentation', 'OCR',
            'OpenCV', 'YOLO', 'ResNet', 'CNN', 'Convolutional Neural Networks',
    
            # Data Science & Analytics
            'Data Science', 'Data Mining', 'Statistical Analysis',
            'Predictive Modeling', 'Feature Engineering', 'Model Evaluation',
            'A/B Testing', 'Hypothesis Testing', 'Time Series Analysis',
    
            # ML Operations
            'MLOps', 'Model Deployment', 'Model Monitoring', 'AutoML',
            'Hyperparameter Tuning', 'Model Optimization', 'Model Serving',
            'MLflow', 'Kubeflow', 'SageMaker', 'Vertex AI',
    
            # Big Data & Processing
            'Big Data', 'Apache Spark', 'PySpark', 'Hadoop', 'Hive',
            'Apache Kafka', 'Apache Flink', 'Databricks',
            'Data Pipeline', 'ETL', 'Data Warehousing',
    
            # Tools & Platforms
            'Jupyter', 'Notebook', 'Google Colab', 'Kaggle',
            'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn', 'Plotly',
    
            # Specialized AI
            'Recommender Systems', 'Anomaly Detection', 'Forecasting',
            'Speech Recognition', 'Voice Assistant', 'Chatbot',
            'Graph Neural Networks', 'GNN', 'Federated Learning',
            'Edge AI', 'Neural Architecture Search', 'NAS',
    
            # Research & Theory
            'Research', 'Algorithm Design', 'Optimization',
            'Linear Algebra', 'Calculus', 'Probability', 'Statistics',
            'Information Theory', 'Bayesian Methods',
        ]
        # Combine all skill libraries
        all_skill_keywords = (
            tech_skills + engineering_skills + standards_skills +
            pm_skills + business_skills + ai_ml_skills
        )
        
        # Case-insensitive matching
        text_lower = text.lower()
        for skill in all_skill_keywords:
            # Word boundary matching
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                skills.add(skill)
        
        # === STRATEGY 3: Pattern-based extraction ===
        
        # 3A. "Experience with X" / "Proficiency in X"
        experience_patterns = [
            r'experience (?:with|in) ([A-Z][A-Za-z\s\-\.#\+]{2,25})(?:\s+and\b|[,;.\n]|$)',
            r'proficien(?:t|cy) in ([A-Z][A-Za-z\s\-\.#\+]{2,25})(?:\s+(?:and|for)\b|[,;.\n]|$)',
            r'knowledge of ([A-Z][A-Za-z\s\-\.#\+]{2,25})(?:\s+and\b|[,;.\n]|$)',
            r'skilled in ([A-Z][A-Za-z\s\-\.#\+]{2,25})(?:\s+and\b|[,;.\n]|$)',
            r'expertise in ([A-Z][A-Za-z\s\-\.#\+]{2,25})(?:\s+and\b|[,;.\n]|$)',
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean = match.strip().rstrip('.,:;')
                if len(clean) > 2:
                    skills.add(clean.title())
        
        # 3B. "X years of Y"
        years_pattern = r'\d+\+?\s+years? (?:of\s+)?(?:experience\s+)?(?:with|in)\s+([A-Z][A-Za-z\s\-]{2,30})'
        years_matches = re.findall(years_pattern, text)
        for match in years_matches:
            clean = match.strip().rstrip('.,:;')
            if len(clean) > 2:
                skills.add(clean.title())
        
        # === STRATEGY 4: Multi-word technical phrases ===
        
        # Common skill phrases
        skill_phrases = [
            # Engineering
            'Process Safety Management', 'Process Hazard Analysis',
            'Risk Assessment', 'Root Cause Analysis', 'Failure Mode Analysis',
            'Safety Instrumented Systems', 'Layers of Protection',
            'Chemical Engineering', 'Mechanical Engineering', 'Electrical Engineering',
            
            # Technical
            'Machine Learning', 'Deep Learning', 'Natural Language Processing',
            'Computer Vision', 'Data Science', 'Data Analysis',
            'Business Intelligence', 'Data Warehousing', 'ETL',
            
            # Project/Quality
            'Project Management', 'Program Management', 'Product Management',
            'Quality Assurance', 'Quality Control', 'Regulatory Compliance',
            'Change Management', 'Stakeholder Management',
            
            # Soft Skills
            'Problem Solving', 'Critical Thinking', 'Analytical Thinking',
            'Team Leadership', 'Communication Skills',
        ]
        
        for phrase in skill_phrases:
            if phrase.lower() in text_lower:
                skills.add(phrase)
        
        # === CLEANUP ===
        cleaned = self._clean_skills(skills)
        
        return sorted(cleaned)
    
    def _extract_ner(self, text: str) -> List[str]:
        """
        NER-based extraction using transformers model
        """
        if not HAS_TRANSFORMERS:
            return []
        
        # Initialize pipeline if needed
        if self._pipeline is None:
            model_name = self._get_best_available_model()
            if model_name is None:
                return []
            
            try:
                print(f"Loading model: {model_name}...")
                self._pipeline = pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple",
                    device=-1
                )
            except Exception as e:
                print(f"Failed to load model: {e}")
                return []
        
        # Run NER
        try:
            # Truncate long text
            if len(text) > 5000:
                text = text[:5000]
            
            entities = self._pipeline(text)
            
            # Extract skills
            skills = []
            for entity in entities:
                label = entity.get('entity_group', '').upper()
                score = entity.get('score', 0)
                
                # Filter by label and confidence
                if ('SKILL' in label or 'KNOWLEDGE' in label or 
                    'HSKILL' in label or 'SSKILL' in label) and score > 0.5:
                    
                    word = entity['word'].strip()
                    # Clean tokenization artifacts
                    word = word.replace(' ##', '').replace('##', '').strip()
                    
                    if len(word) > 1:
                        skills.append(word)
            
            return skills[:50]
            
        except Exception as e:
            print(f"NER extraction failed: {e}")
            return []
    
    def _get_best_available_model(self) -> Optional[str]:
        """Try to find the best available model"""
        if self.use_model == "auto":
            # Try models in priority order
            for model_key in ["jobberta", "nucha", "distilbert", "jobspanbert"]:
                model_name = self.model_priority[model_key]
                try:
                    # Test if model is accessible
                    from transformers import AutoModel
                    AutoModel.from_pretrained(model_name)
                    print(f"Using model: {model_name}")
                    return model_name
                except:
                    continue
            return None
        else:
            return self.model_priority.get(self.use_model)
    
    def _clean_skills(self, skills: Set[str]) -> List[str]:
        """Clean and filter extracted skills"""
        cleaned = []
        
        # Noise patterns
        noise = {
            'the', 'and', 'or', 'for', 'with', 'from', 'about',
            'job', 'work', 'team', 'role', 'position',
            'required', 'preferred', 'must', 'should',
        }
        
        for skill in skills:
            skill = skill.strip()
            
            # Skip if too short or too long
            if len(skill) < 2 or len(skill) > 60:
                continue
            
            # Skip if it's just a noise word
            if skill.lower() in noise:
                continue
            
            # Skip if starts with noise words
            if any(skill.lower().startswith(n + ' ') for n in noise):
                continue
            
            cleaned.append(skill)
        
        # Remove duplicates (case-insensitive)
        unique = []
        seen = set()
        for skill in cleaned:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                unique.append(skill)
        
        return unique


def extract_jd_from_text(text: str, strategy: str = "hybrid") -> Dict:
    """
    Main interface: Extract skills from JD text
    
    Args:
        text: Job description text
        strategy: "regex", "ner", or "hybrid" (default)
    
    Returns:
        Dict with extracted skills and metadata
    """
    extractor = SkillExtractor()
    skills = extractor.extract_skills(text, strategy=strategy)
    
    # Extract sections (simple heuristic)
    sections = []
    if 'responsibilities' in text.lower():
        sections.append('Responsibilities')
    if 'qualifications' in text.lower() or 'requirements' in text.lower():
        sections.append('Requirements')
    
    return {
        'skills': skills,
        'skill_count': len(skills),
        'sections': sections,
        'success': True,
        'strategy_used': strategy
    }


def extract_jd_from_url(url: str, strategy: str = "hybrid") -> Dict:
    """Extract JD from URL"""
    if not HAS_WEB:
        return {
            'skills': [],
            'success': False,
            'error': 'requests/beautifulsoup not available'
        }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Process text
        result = extract_jd_from_text(text, strategy=strategy)
        result['source_url'] = url
        return result
        
    except Exception as e:
        return {
            'skills': [],
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test with Process Safety Engineer JD
    test_jd = """
    ob Description
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
    
    print("\n" + "="*80)
    print("NEW ALGORITHM TEST")
    print("="*80)
    
    result = extract_jd_from_text(test_jd, strategy="regex")
    
    print(f"\n✅ Extracted {result['skill_count']} skills:")
    for i, skill in enumerate(result['skills'], 1):
        print(f"  {i:2d}. {skill}")
    
    print(f"\nStrategy: {result['strategy_used']}")
    print("="*80)