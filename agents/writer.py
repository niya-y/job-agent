"""
Cover Letter Writer using Llama-3.1-8B-Instruct
Generates professional, ATS-friendly cover letters for international jobs
"""

from typing import List, Dict, Optional
from huggingface_hub import InferenceClient


# Professional system prompt for Llama-3.1
SYSTEM_PROMPT = """You are an expert career coach specializing in international job applications. Your task is to write professional, compelling cover letters that help candidates stand out to hiring managers at foreign companies.

STYLE GUIDELINES:
- Professional but warm and authentic tone
- Active voice and strong action verbs
- Specific achievements with quantifiable metrics
- Natural integration of required keywords
- ATS-friendly formatting (no special characters)
- 350-450 words (concise and impactful)

STRUCTURE:
1. Opening: Express interest and mention how you learned about the role
2. Body (2-3 paragraphs):
   - Highlight 2-3 most relevant experiences
   - Quantify achievements where possible
   - Show understanding of company/role
3. Closing: Express enthusiasm and call to action

WHAT TO AVOID:
- Generic phrases like "I am a team player"
- Overly formal or stiff language
- Simply repeating resume content
- Spelling or grammar errors
- Overused adjectives (passionate, motivated, etc.)

EXAMPLES OF GOOD PHRASES:
✓ "Reduced processing time by 40% through optimized ETL workflows"
✓ "Led a cross-functional team of 5 engineers to deliver..."
✓ "Implemented a real-time data pipeline handling 10M+ events daily"
✗ "I am passionate about data engineering"
✗ "I have excellent communication skills"

Remember: Show, don't tell. Use concrete examples instead of abstract claims."""


USER_TEMPLATE = """Write a professional cover letter for the following job application.

JOB DESCRIPTION SUMMARY:
{jd_summary}

REQUIRED SKILLS (from job posting):
{jd_skills}

CANDIDATE'S RELEVANT EXPERIENCES:
{matched_experiences}

INSTRUCTIONS:
- Write in first person from the candidate's perspective
- Integrate the required skills naturally
- Use specific examples from the matched experiences
- Keep it under 450 words
- Make it compelling and authentic
- DO NOT use placeholder text like [Company Name] or [Your Name]
- Write as if you are the candidate applying

Generate only the cover letter text, no additional commentary."""


class CoverLetterWriter:
    """
    Generate professional cover letters using HuggingFace LLMs
    """
    
    def __init__(
        self,
        model: str = "mHuggingFaceH4/zephyr-7b-beta",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800
    ):
        """
        Initialize Cover Letter Writer
        
        Args:
            model: HuggingFace model identifier
            api_key: HuggingFace API token
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum output length
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize HuggingFace client
        try:
            self.client = InferenceClient(model=model, token=api_key)
            print(f"✓ Initialized {model}")
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            self.client = None
    
    def generate_cover_letter(
        self,
        jd_data: Dict,
        matches: List[Dict]
    ) -> Dict:
        """
        Generate cover letter from JD and matched experiences
        
        Args:
            jd_data: Output from jd_extractor
            matches: Output from matcher.search()
            
        Returns:
            Dict with 'cover_letter' and metadata
        """
        # Prepare inputs
        jd_summary = self._prepare_jd_summary(jd_data)
        jd_skills = self._prepare_skills_list(jd_data)
        matched_exp = self._prepare_experiences(matches)
        
        # Format prompt
        user_prompt = USER_TEMPLATE.format(
            jd_summary=jd_summary,
            jd_skills=jd_skills,
            matched_experiences=matched_exp
        )
        
        # Generate
        try:
            cover_letter = self._generate_with_llm(user_prompt)
            
            return {
                "cover_letter": cover_letter,
                "word_count": len(cover_letter.split()),
                "model_used": self.model,
                "success": True
            }
            
        except Exception as e:
            print(f"Error generating cover letter: {e}")
            # Fallback to template
            return self._generate_fallback(jd_data, matches)
    
    def _generate_with_llm(self, user_prompt: str) -> str:
        """Call HuggingFace Inference API"""
        if self.client is None:
            raise RuntimeError("Model not initialized")
        
        # Format chat messages for Llama-3.1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate with streaming disabled for simplicity
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            stream=False
        )
        
        # Extract generated text
        if hasattr(response, 'choices') and len(response.choices) > 0:
            cover_letter = response.choices[0].message.content
        else:
            # Fallback for different response formats
            cover_letter = str(response)
        
        return cover_letter.strip()
    
    def _prepare_jd_summary(self, jd_data: Dict) -> str:
        """Create concise JD summary"""
        # Use first section if available
        if jd_data.get("sections"):
            summary = jd_data["sections"][0]
            # Truncate if too long
            if len(summary) > 1500:
                summary = summary[:1500] + "..."
            return summary
        
        # Fallback to clean text
        text = jd_data.get("clean_text", "")
        if len(text) > 1500:
            text = text[:1500] + "..."
        return text
    
    def _prepare_skills_list(self, jd_data: Dict) -> str:
        """Format skills as bullet points"""
        skills = jd_data.get("skills", [])
        
        if not skills:
            skills = jd_data.get("keywords", [])[:20]
        
        if not skills:
            return "Skills not explicitly listed"
        
        # Format as list
        skills_str = "\n".join([f"- {skill}" for skill in skills[:25]])
        return skills_str
    
    def _prepare_experiences(self, matches: List[Dict]) -> str:
        """Format matched experiences"""
        if not matches:
            return "No specific experiences provided"
        
        # Take top matches
        top_matches = matches[:5]
        
        experiences = []
        for i, match in enumerate(top_matches, 1):
            score = match.get("score", 0)
            text = match.get("text", "")
            
            # Clean up formatting
            text = text.replace(" | ", "\n")
            
            experiences.append(f"{i}. (Relevance: {score:.2f})\n{text}")
        
        return "\n\n".join(experiences)
    
    def _generate_fallback(self, jd_data: Dict, matches: List[Dict]) -> Dict:
        """
        Simple template-based fallback when LLM fails
        """
        skills = jd_data.get("skills", jd_data.get("keywords", []))
        skills_str = ", ".join(skills[:10]) if skills else "the required skills"
        
        experiences = "\n".join([
            f"• {match['text'].split('|')[0]}" 
            for match in matches[:3]
        ]) if matches else "No matching experiences"
        
        template = f"""Dear Hiring Manager,

I am writing to express my interest in this position. My background in {skills_str} aligns well with your requirements.

Key relevant experiences:
{experiences}

I am excited about the opportunity to contribute to your team and would welcome the chance to discuss how my skills and experience can benefit your organization.

Thank you for considering my application.

Best regards"""
        
        return {
            "cover_letter": template,
            "word_count": len(template.split()),
            "model_used": "fallback_template",
            "success": False,
            "note": "Used fallback template due to LLM error"
        }
    
    def generate_multiple_versions(
        self,
        jd_data: Dict,
        matches: List[Dict],
        num_versions: int = 3
    ) -> List[Dict]:
        """
        Generate multiple versions with slight temperature variations
        
        Useful for giving candidates options to choose from
        """
        versions = []
        
        for i in range(num_versions):
            # Vary temperature slightly
            original_temp = self.temperature
            self.temperature = max(0.5, min(1.0, original_temp + (i - 1) * 0.1))
            
            version = self.generate_cover_letter(jd_data, matches)
            version['version'] = i + 1
            versions.append(version)
            
            # Restore original temperature
            self.temperature = original_temp
        
        return versions


# Convenience function
def write_cover_letter(
    jd_data: Dict,
    matches: List[Dict],
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key: Optional[str] = None
) -> str:
    """
    Quick function to generate a cover letter
    
    Returns just the cover letter text
    """
    writer = CoverLetterWriter(model=model, api_key=api_key)
    result = writer.generate_cover_letter(jd_data, matches)
    return result.get("cover_letter", "")


if __name__ == "__main__":
    # Test example
    sample_jd = {
        "skills": ["Python", "AWS", "Data Engineering", "ETL"],
        "sections": ["Looking for experienced Data Engineer with Python and AWS"]
    }
    
    sample_matches = [
        {
            "score": 0.85,
            "text": "Company: Tech Corp | Role: Data Engineer | Built ETL pipelines"
        }
    ]
    
    print("Testing cover letter generation...")
    # Would need HF token
    # letter = write_cover_letter(sample_jd, sample_matches)
    # print(letter)