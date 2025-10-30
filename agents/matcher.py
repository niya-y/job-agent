"""
Resume-JD Matcher using Snowflake Arctic Embed
Performs semantic search to find relevant experiences matching job requirements
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None
    print("Warning: faiss-cpu not installed. Install with: pip install faiss-cpu")

from utils.embedding import EmbeddingClient
from utils.parsing import chunk_texts


class ResumeMatcher:
    """
    Matches resume experiences to job descriptions using vector similarity
    """
    
    def __init__(
        self,
        embed_provider: str = "huggingface",
        embed_model: str = "Snowflake/snowflake-arctic-embed-m",
        api_key: Optional[str] = None,
        index_dir: str = "vectorstore",
        match_threshold: float = 0.35
    ):
        """
        Initialize Resume Matcher
        
        Args:
            embed_provider: Provider name (huggingface, openai)
            embed_model: Model identifier
            api_key: API key (HuggingFace token)
            index_dir: Directory to store FAISS index
            match_threshold: Minimum similarity score (0.0-1.0)
        """
        self.embed_provider = embed_provider
        self.embed_model = embed_model
        self.match_threshold = match_threshold
        
        # Initialize embedding client
        self.client = EmbeddingClient(
            provider=embed_provider,
            model=embed_model,
            api_key=api_key
        )
        
        # Set up index storage
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.index_path = self.index_dir / "resume_index.faiss"
        self.meta_path = self.index_dir / "resume_meta.json"
        
        self.index = None
        self.metadata = []
    
    def build_index_from_resume(self, resume_data: Dict) -> bool:
        """
        Build FAISS index from resume data
        
        Args:
            resume_data: Resume JSON with structure:
                {
                    "experiences": [
                        {
                            "company": "Tech Corp",
                            "title": "Software Engineer",
                            "description": "Built scalable systems..."
                        }
                    ]
                }
        
        Returns:
            True if successful
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        
        # Extract and format experiences
        experiences = self._extract_experiences(resume_data)
        
        if not experiences:
            raise ValueError(
                "No experiences found in resume data. "
                "Check that resume.json has 'experiences' field."
            )
        
        print(f"Processing {len(experiences)} experience entries...")
        
        # Chunk long experiences
        chunks = chunk_texts(experiences, chunk_size=400, overlap=50)
        
        print(f"Created {len(chunks)} chunks for indexing...")
        
        # Generate embeddings
        print("Generating embeddings with Snowflake Arctic Embed...")
        embeddings = self.client.embed(chunks)
        
        # Create FAISS index
        dim = len(embeddings[0])
        print(f"Building FAISS index (dimension: {dim})...")
        
        # Use Inner Product (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dim)
        
        # Normalize vectors for cosine similarity
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        
        # Add to index
        index.add(vecs)
        
        self.index = index
        self.metadata = chunks
        
        # Save to disk
        self._save_index()
        
        print(f"✓ Index built successfully with {len(chunks)} vectors")
        return True
    
    def _extract_experiences(self, resume_data: Dict) -> List[str]:
        """
        Extract and format experience entries
        
        Returns formatted strings like:
        "Company: Tech Corp | Role: Senior Engineer | Details: Built systems..."
        """
        experiences = []
        
        for exp in resume_data.get("experiences", []):
            parts = []
            
            # Add company
            if exp.get("company"):
                parts.append(f"Company: {exp['company']}")
            
            # Add title/role
            if exp.get("title"):
                parts.append(f"Role: {exp['title']}")
            
            # Add description
            description = exp.get("description", "").strip()
            if description:
                parts.append(f"Details: {description}")
            
            # Combine
            if parts:
                experience_text = " | ".join(parts)
                experiences.append(experience_text)
        
        return experiences
    
    def search(
        self,
        query: str,
        top_k: int = 6
    ) -> List[Dict]:
        """
        Search for relevant experiences
        
        Args:
            query: Search query (JD keywords or full JD section)
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'score' and 'text' keys, sorted by relevance
        """
        if self.index is None:
            # Try to load from disk
            if not self._load_index():
                raise RuntimeError(
                    "No index found. Please build index first with "
                    "build_index_from_resume()"
                )
        
        # Generate query embedding
        query_emb = self.client.embed([query])[0]
        
        # Normalize for cosine similarity
        q = np.array([query_emb], dtype="float32")
        faiss.normalize_L2(q)
        
        # Search
        distances, indices = self.index.search(q, top_k)
        
        # Format results
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result
                continue
            
            # Apply threshold filter
            if score < self.match_threshold:
                continue
            
            results.append({
                "score": float(score),
                "text": self.metadata[idx]
            })
        
        return results
    
    def search_by_skills(
        self,
        skills: List[str],
        top_k: int = 6
    ) -> List[Dict]:
        """
        Search using skill keywords
        
        Args:
            skills: List of skill keywords from JD
            top_k: Number of results
            
        Returns:
            List of matching experiences
        """
        # Create natural language query
        if len(skills) > 30:
            skills = skills[:30]  # Limit for token length
        
        # Format as professional query
        query = (
            f"Professional experience with: {', '.join(skills[:15])}. "
            f"Additional skills: {', '.join(skills[15:25]) if len(skills) > 15 else 'N/A'}"
        )
        
        return self.search(query, top_k)
    
    def search_by_jd_section(
        self,
        jd_section: str,
        top_k: int = 6
    ) -> List[Dict]:
        """
        Search using full JD section (Requirements, Responsibilities)
        
        Args:
            jd_section: Full text of JD section
            top_k: Number of results
            
        Returns:
            List of matching experiences
        """
        # Truncate if too long
        max_length = 2000
        if len(jd_section) > max_length:
            jd_section = jd_section[:max_length]
        
        return self.search(jd_section, top_k)
    
    def _save_index(self):
        """Save index and metadata to disk"""
        if self.index is None:
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Index saved to {self.index_dir}")
            
        except Exception as e:
            print(f"Warning: Could not save index: {e}")
    
    def _load_index(self) -> bool:
        """Load index from disk"""
        try:
            if not self.index_path.exists() or not self.meta_path.exists():
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            print(f"✓ Index loaded from {self.index_dir}")
            return True
            
        except Exception as e:
            print(f"Warning: Could not load index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {"status": "not_built"}
        
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "embedding_model": self.embed_model,
            "threshold": self.match_threshold
        }


# Convenience function for quick matching
def match_resume_to_jd(
    resume_data: Dict,
    jd_data: Dict,
    embed_model: str = "Snowflake/snowflake-arctic-embed-m",
    api_key: Optional[str] = None,
    top_k: int = 6
) -> List[Dict]:
    """
    One-shot matching function
    
    Args:
        resume_data: Resume JSON
        jd_data: JD extraction result (from jd_extractor)
        embed_model: Embedding model name
        api_key: HuggingFace token
        top_k: Number of matches
        
    Returns:
        List of matching experiences with scores
    """
    # Initialize matcher
    matcher = ResumeMatcher(
        embed_provider="huggingface",
        embed_model=embed_model,
        api_key=api_key
    )
    
    # Build index
    matcher.build_index_from_resume(resume_data)
    
    # Use skills first, fallback to sections
    if jd_data.get("skills"):
        matches = matcher.search_by_skills(jd_data["skills"], top_k)
    elif jd_data.get("sections"):
        # Use first section (usually requirements)
        matches = matcher.search_by_jd_section(jd_data["sections"][0], top_k)
    else:
        # Fallback to full text
        matches = matcher.search(jd_data.get("clean_text", ""), top_k)
    
    return matches


if __name__ == "__main__":
    # Test example
    sample_resume = {
        "experiences": [
            {
                "company": "Tech Corp",
                "title": "Senior Data Engineer",
                "description": "Built ETL pipelines using Python and Spark. Optimized AWS infrastructure."
            },
            {
                "company": "Startup Inc",
                "title": "Software Engineer",
                "description": "Developed REST APIs with Node.js and PostgreSQL. Implemented CI/CD."
            }
        ]
    }
    
    sample_jd = {
        "skills": ["Python", "Spark", "AWS", "ETL"],
        "sections": ["Experience with big data technologies required"]
    }
    
    print("Testing matcher...")
    # Would need HF token in environment
    # matches = match_resume_to_jd(sample_resume, sample_jd)
    # print(f"Found {len(matches)} matches")