"""
Resume-JD Matching Analyzer
Analyzes compatibility between resume and job description
"""

from typing import Dict, List, Any, Optional
from collections import Counter
import re


class MatchingAnalyzer:
    """Analyze resume-JD compatibility and generate insights"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'sql', 'r', 'scala', 'go'],
            'data': ['spark', 'hadoop', 'kafka', 'airflow', 'etl', 'data pipeline', 'data warehouse'],
            'cloud': ['aws', 's3', 'ec2', 'lambda', 'redshift', 'glue', 'azure', 'gcp'],
            'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'mlops'],
            'tools': ['docker', 'kubernetes', 'git', 'jenkins', 'ci/cd', 'terraform'],
            'database': ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb']
        }
    
    def analyze(
        self,
        jd_data: Dict[str, Any],
        matches: List[Dict[str, Any]],
        resume_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of resume-JD matching
        
        Args:
            jd_data: Extracted job description data
            matches: Top matching experiences from FAISS search
            resume_data: Full resume data (optional)
        
        Returns:
            Analysis report with scores, insights, and recommendations
        """
        # Extract skills from JD
        jd_skills = self._normalize_skills(jd_data.get('skills', []))
        
        # Extract skills from matched experiences
        matched_skills = self._extract_skills_from_matches(matches)
        
        # Calculate matching scores
        skill_match = self._calculate_skill_match(jd_skills, matched_skills)
        experience_relevance = self._calculate_experience_relevance(matches)
        
        # Overall compatibility score (weighted average)
        overall_score = (skill_match['score'] * 0.6) + (experience_relevance['score'] * 0.4)
        
        # Generate insights
        insights = self._generate_insights(
            jd_skills, 
            matched_skills, 
            skill_match, 
            overall_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            skill_match, 
            experience_relevance,
            overall_score
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'compatibility_level': self._get_compatibility_level(overall_score),
            'skill_match': skill_match,
            'experience_relevance': experience_relevance,
            'insights': insights,
            'recommendations': recommendations,
            'top_strengths': self._identify_top_strengths(matched_skills, jd_skills),
            'gaps': self._identify_gaps(matched_skills, jd_skills)
        }
    
    def _normalize_skills(self, skills: List[str]) -> set:
        """Normalize skill names for comparison"""
        normalized = set()
        for skill in skills:
            # Lowercase and remove extra spaces
            skill_clean = ' '.join(skill.lower().strip().split())
            normalized.add(skill_clean)
        return normalized
    
    def _extract_skills_from_matches(self, matches: List[Dict]) -> set:
        """Extract skills mentioned in matched experiences"""
        skills = set()
        for match in matches:
            text = match.get('text', '').lower()
            
            # Extract from all skill categories
            for category, skill_list in self.skill_categories.items():
                for skill in skill_list:
                    if skill in text or skill.replace(' ', '-') in text:
                        skills.add(skill)
            
            # Extract from metadata if available
            if 'skills' in match:
                skills.update(self._normalize_skills(match['skills']))
        
        return skills
    
    def _calculate_skill_match(
        self, 
        jd_skills: set, 
        matched_skills: set
    ) -> Dict[str, Any]:
        """Calculate skill matching percentage"""
        if not jd_skills:
            return {
                'score': 0,
                'matched': [],
                'missing': [],
                'percentage': 0
            }
        
        # Find matched and missing skills
        matched = jd_skills & matched_skills
        missing = jd_skills - matched_skills
        
        # Calculate percentage
        percentage = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0
        
        # Score (0-100, with bonus for extra skills)
        base_score = percentage
        extra_skills = len(matched_skills - jd_skills)
        bonus = min(extra_skills * 2, 20)  # Max 20% bonus
        score = min(base_score + bonus, 100)
        
        return {
            'score': round(score, 2),
            'matched': sorted(list(matched)),
            'missing': sorted(list(missing)),
            'percentage': round(percentage, 2),
            'extra_skills': sorted(list(matched_skills - jd_skills))
        }
    
    def _calculate_experience_relevance(self, matches: List[Dict]) -> Dict[str, Any]:
        """Calculate how relevant the experiences are"""
        if not matches:
            return {
                'score': 0,
                'top_match_score': 0,
                'avg_score': 0,
                'strong_matches': 0
            }
        
        # Extract similarity scores
        scores = [m.get('score', 0) for m in matches]
        
        # Calculate metrics
        top_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        strong_matches = sum(1 for s in scores if s >= 0.7)
        
        # Overall relevance score (0-100)
        # Based on: top match quality + average quality + number of strong matches
        relevance_score = (
            (top_score * 40) +           # Top match: 40%
            (avg_score * 40) +           # Average: 40%
            (min(strong_matches / 3, 1) * 20)  # Strong matches: 20%
        ) * 100
        
        return {
            'score': round(relevance_score, 2),
            'top_match_score': round(top_score, 3),
            'avg_score': round(avg_score, 3),
            'strong_matches': strong_matches,
            'total_matches': len(matches)
        }
    
    def _generate_insights(
        self,
        jd_skills: set,
        matched_skills: set,
        skill_match: Dict,
        overall_score: float
    ) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # Overall assessment
        if overall_score >= 80:
            insights.append("🎯 Excellent match! Your profile strongly aligns with the job requirements.")
        elif overall_score >= 60:
            insights.append("✅ Good match. You meet most of the key requirements with some gaps.")
        elif overall_score >= 40:
            insights.append("⚠️ Moderate match. You have relevant experience but several skill gaps exist.")
        else:
            insights.append("❌ Limited match. Consider whether this role is the right fit.")
        
        # Skill coverage
        if skill_match['percentage'] >= 70:
            insights.append(f"💪 Strong skill coverage: {skill_match['percentage']:.0f}% of required skills matched.")
        elif skill_match['percentage'] >= 50:
            insights.append(f"📊 Decent skill coverage: {skill_match['percentage']:.0f}% of required skills matched.")
        else:
            insights.append(f"📉 Low skill coverage: Only {skill_match['percentage']:.0f}% of required skills matched.")
        
        # Bonus skills
        if skill_match.get('extra_skills'):
            extra_count = len(skill_match['extra_skills'])
            insights.append(f"⭐ You bring {extra_count} additional skill(s) beyond the job requirements.")
        
        # Missing critical skills
        if len(skill_match['missing']) > 0:
            critical_missing = [s for s in skill_match['missing'] 
                              if any(cat_skill in s for cat in ['programming', 'cloud', 'data'] 
                                   for cat_skill in self.skill_categories.get(cat, []))]
            if critical_missing and len(critical_missing) <= 3:
                insights.append(f"🔴 Critical gaps: {', '.join(critical_missing[:3])}")
        
        return insights
    
    def _generate_recommendations(
        self,
        skill_match: Dict,
        experience_relevance: Dict,
        overall_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on overall score
        if overall_score >= 70:
            recommendations.append("✍️ Highlight your matching skills prominently in your cover letter.")
            recommendations.append("📝 Use specific examples from your top matching experiences.")
        else:
            recommendations.append("🎓 Consider upskilling in the missing critical areas before applying.")
            recommendations.append("💡 Focus on transferable skills in your application.")
        
        # Skill-based recommendations
        if skill_match['missing']:
            top_missing = skill_match['missing'][:3]
            recommendations.append(f"📚 Priority skills to learn: {', '.join(top_missing)}")
        
        # Experience-based recommendations
        if experience_relevance['strong_matches'] >= 3:
            recommendations.append("🌟 You have multiple strong matching experiences - feature them prominently!")
        elif experience_relevance['strong_matches'] == 0:
            recommendations.append("⚡ Consider reframing your experiences to better highlight relevant aspects.")
        
        # Cover letter strategy
        if skill_match['percentage'] < 60:
            recommendations.append("💬 In your cover letter, emphasize your ability to learn quickly and adapt.")
        else:
            recommendations.append("💬 In your cover letter, demonstrate how your skills solve their specific challenges.")
        
        return recommendations
    
    def _identify_top_strengths(self, matched_skills: set, jd_skills: set) -> List[str]:
        """Identify top 5 strengths"""
        # Prioritize matched JD skills
        matched = list(jd_skills & matched_skills)
        
        # Add extra skills
        extra = list(matched_skills - jd_skills)
        
        # Combine and limit to top 5
        strengths = matched + extra
        return strengths[:5]
    
    def _identify_gaps(self, matched_skills: set, jd_skills: set) -> List[str]:
        """Identify skill gaps (up to 5)"""
        gaps = list(jd_skills - matched_skills)
        return gaps[:5]
    
    def _get_compatibility_level(self, score: float) -> str:
        """Convert score to compatibility level"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Moderate"
        else:
            return "Limited"
    
    def generate_text_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted text report"""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 RESUME-JD MATCHING ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Overall score
        lines.append(f"Overall Compatibility: {analysis['overall_score']}/100 ({analysis['compatibility_level']})")
        lines.append("")
        
        # Breakdown
        lines.append("Detailed Breakdown:")
        lines.append(f"  • Skill Match: {analysis['skill_match']['score']:.1f}/100 ({analysis['skill_match']['percentage']:.0f}% coverage)")
        lines.append(f"  • Experience Relevance: {analysis['experience_relevance']['score']:.1f}/100")
        lines.append("")
        
        # Strengths
        if analysis['top_strengths']:
            lines.append("Top Strengths:")
            for strength in analysis['top_strengths']:
                lines.append(f"  ✓ {strength}")
            lines.append("")
        
        # Gaps
        if analysis['gaps']:
            lines.append("Skill Gaps:")
            for gap in analysis['gaps']:
                lines.append(f"  ✗ {gap}")
            lines.append("")
        
        # Insights
        lines.append("Key Insights:")
        for insight in analysis['insights']:
            lines.append(f"  {insight}")
        lines.append("")
        
        # Recommendations
        lines.append("Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)