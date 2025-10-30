"""
Text parsing and chunking utilities
"""

import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text for processing
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip
    return text.strip()


def html_to_text(html: str) -> str:
    """
    Convert HTML to plain text
    
    Args:
        html: HTML string
        
    Returns:
        Plain text
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        return normalize_text(text)
        
    except ImportError:
        # Fallback: simple regex
        text = re.sub(r'<[^>]+>', '', html)
        return normalize_text(text)


def chunk_texts(
    texts: List[str],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[str]:
    """
    Split long texts into overlapping chunks
    
    Args:
        texts: List of text strings
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    for text in texts:
        if len(text) <= chunk_size:
            # Short enough, keep as is
            chunks.append(text)
        else:
            # Split into chunks
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for last period, exclamation, or question mark
                    last_break = max(
                        chunk.rfind('. '),
                        chunk.rfind('! '),
                        chunk.rfind('? ')
                    )
                    
                    if last_break > chunk_size * 0.5:  # At least 50% into chunk
                        chunk = chunk[:last_break + 1]
                        end = start + last_break + 1
                
                chunks.append(chunk.strip())
                
                # Move to next chunk with overlap
                start = end - overlap
    
    return chunks


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum characters
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text
    
    Finds lines starting with -, •, *, or numbers
    
    Args:
        text: Text containing bullet points
        
    Returns:
        List of bullet point contents
    """
    lines = text.split('\n')
    bullets = []
    
    for line in lines:
        line = line.strip()
        
        # Match bullet patterns
        match = re.match(r'^[-•*]\s+(.+)$', line)
        if match:
            bullets.append(match.group(1))
            continue
        
        # Match numbered lists
        match = re.match(r'^\d+[.)]\s+(.+)$', line)
        if match:
            bullets.append(match.group(1))
    
    return bullets


def clean_company_name(company: str) -> str:
    """
    Clean company name (remove Inc, LLC, etc.)
    
    Args:
        company: Raw company name
        
    Returns:
        Cleaned name
    """
    # Remove common suffixes
    suffixes = [
        r',?\s+Inc\.?',
        r',?\s+LLC\.?',
        r',?\s+Ltd\.?',
        r',?\s+Corp\.?',
        r',?\s+Corporation',
        r',?\s+Company',
        r',?\s+Co\.?'
    ]
    
    cleaned = company
    for suffix in suffixes:
        cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text
    
    Args:
        text: Text containing URLs
        
    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text
    
    Args:
        text: Text containing emails
        
    Returns:
        List of email addresses
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def count_words(text: str) -> int:
    """
    Count words in text
    
    Args:
        text: Text to count
        
    Returns:
        Word count
    """
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences
    
    Simple rule-based splitter
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Split on period, exclamation, question mark followed by space
    sentences = re.split(r'[.!?]\s+', text)
    
    # Clean up
    return [s.strip() for s in sentences if s.strip()]


def format_experience_text(
    company: str,
    title: str,
    description: str
) -> str:
    """
    Format experience entry into consistent text
    
    Args:
        company: Company name
        title: Job title
        description: Job description
        
    Returns:
        Formatted text
    """
    parts = []
    
    if company:
        parts.append(f"Company: {clean_company_name(company)}")
    
    if title:
        parts.append(f"Role: {title}")
    
    if description:
        parts.append(f"Details: {description}")
    
    return " | ".join(parts)


if __name__ == "__main__":
    # Test
    sample = "I worked at Tech Corp, Inc. building ETL pipelines with Python and Spark. " * 20
    
    chunks = chunk_texts([sample], chunk_size=100)
    print(f"Split into {len(chunks)} chunks")
    
    print(f"Word count: {count_words(sample)}")