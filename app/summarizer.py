"""
Content Summarizer component for generating refined text from sections
"""

import logging
import re
from typing import List, Dict
from collections import Counter

class ContentSummarizer:
    """Summarizes and refines content sections"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def summarize_section(self, content: str, query_context: str, 
                         max_length: int = 500) -> str:
        """
        Summarize a section's content focusing on relevance to query context
        
        Args:
            content: Original section content
            query_context: Persona and job context for relevance focus
            max_length: Maximum length of summary in characters
            
        Returns:
            Refined/summarized content
        """
        try:
            if not content or not content.strip():
                return ""
            
            # If content is already short enough, clean and return
            if len(content) <= max_length:
                return self._clean_text(content)
            
            # Extract key sentences based on relevance
            key_sentences = self._extract_key_sentences(content, query_context)
            
            # Combine sentences up to max_length
            summary = self._combine_sentences(key_sentences, max_length)
            
            # Final cleaning
            summary = self._clean_text(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing section: {str(e)}")
            # Fallback: return truncated original content
            return self._clean_text(content[:max_length])
    
    def _extract_key_sentences(self, content: str, query_context: str, 
                              max_sentences: int = 10) -> List[str]:
        """
        Extract the most relevant sentences from content
        
        Args:
            content: Original content
            query_context: Query context for relevance scoring
            max_sentences: Maximum number of sentences to extract
            
        Returns:
            List of key sentences
        """
        try:
            # Split into sentences
            sentences = self._split_into_sentences(content)
            
            if len(sentences) <= max_sentences:
                return sentences
            
            # Score sentences by relevance
            scored_sentences = []
            query_words = set(query_context.lower().split())
            
            for sentence in sentences:
                score = self._score_sentence_relevance(sentence, query_words)
                scored_sentences.append((sentence, score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            key_sentences = [sent for sent, score in scored_sentences[:max_sentences]]
            
            return key_sentences
            
        except Exception as e:
            self.logger.error(f"Error extracting key sentences: {str(e)}")
            return self._split_into_sentences(content)[:max_sentences]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # At least 3 words
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _score_sentence_relevance(self, sentence: str, query_words: set) -> float:
        """
        Score a sentence's relevance to the query context
        
        Args:
            sentence: Sentence to score
            query_words: Set of query keywords
            
        Returns:
            Relevance score
        """
        try:
            sentence_words = set(sentence.lower().split())
            
            # Word overlap score
            common_words = query_words.intersection(sentence_words)
            if query_words:
                word_overlap = len(common_words) / len(query_words)
            else:
                word_overlap = 0.0
            
            # Sentence quality factors
            length_score = min(1.0, len(sentence.split()) / 20.0)  # Prefer moderate length
            
            # Check for informative content
            has_numbers = bool(re.search(r'\d+', sentence))
            has_specific_info = bool(re.search(r'[A-Z][a-z]+\s[A-Z][a-z]+', sentence))
            
            info_bonus = 0.1 * (has_numbers + has_specific_info)
            
            # Combine scores
            total_score = word_overlap * 0.7 + length_score * 0.2 + info_bonus
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error scoring sentence: {str(e)}")
            return 0.0
    
    def _combine_sentences(self, sentences: List[str], max_length: int) -> str:
        """
        Combine sentences up to maximum length
        
        Args:
            sentences: List of sentences to combine
            max_length: Maximum character length
            
        Returns:
            Combined text
        """
        combined = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            if len(combined + " " + sentence) > max_length:
                break
            
            if combined:
                combined += " " + sentence
            else:
                combined = sentence
        
        return combined
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and format text
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove incomplete sentences at the end
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            # Find last complete sentence
            last_punct = max(
                cleaned.rfind('.'),
                cleaned.rfind('!'),
                cleaned.rfind('?')
            )
            if last_punct > len(cleaned) * 0.5:  # Only if we're not cutting too much
                cleaned = cleaned[:last_punct + 1]
        
        # Ensure proper capitalization
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned.strip()
    
    def extract_key_points(self, content: str, max_points: int = 5) -> List[str]:
        """
        Extract key points/highlights from content
        
        Args:
            content: Content to analyze
            max_points: Maximum number of points to extract
            
        Returns:
            List of key points
        """
        try:
            # Look for structured content (lists, bullet points)
            points = []
            
            # Find numbered lists
            numbered_items = re.findall(r'\d+\.?\s+([^.]+[.])', content)
            points.extend(numbered_items[:max_points])
            
            if len(points) < max_points:
                # Find bullet points
                bullet_items = re.findall(r'[•\-]\s+([^.]+[.])', content)
                points.extend(bullet_items[:max_points - len(points)])
            
            if len(points) < max_points:
                # Extract key sentences if no structured lists
                sentences = self._split_into_sentences(content)
                remaining = max_points - len(points)
                points.extend(sentences[:remaining])
            
            # Clean points
            cleaned_points = [self._clean_text(point) for point in points if point.strip()]
            
            return cleaned_points[:max_points]
            
        except Exception as e:
            self.logger.error(f"Error extracting key points: {str(e)}")
            return []
    
    def get_content_statistics(self, content: str) -> Dict:
        """
        Get statistics about content
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary with content statistics
        """
        if not content:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "has_structure": False
            }
        
        words = content.split()
        sentences = self._split_into_sentences(content)
        
        # Check for structured content
        has_structure = any(marker in content for marker in 
                          [':', '-', '•', '1.', '2.', 'a)', 'b)', '\n-', '\n•'])
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "has_structure": has_structure
        } 