"""
Relevance Ranker component for scoring and ranking document sections
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from vectorizer import DocumentVectorizer

class RelevanceRanker:
    """Ranks document sections based on relevance to persona and job-to-be-done"""
    
    def __init__(self, vectorizer: DocumentVectorizer):
        """
        Initialize the ranker with a vectorizer
        
        Args:
            vectorizer: DocumentVectorizer instance for embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.vectorizer = vectorizer
    
    def rank_sections(self, parsed_documents: Dict, query_context: str, 
                     max_sections: int = 50) -> List[Dict]:
        """
        Rank all sections from all documents based on relevance to query context
        
        Args:
            parsed_documents: Dictionary of parsed documents
            query_context: Combined persona and job-to-be-done context
            max_sections: Maximum number of sections to consider
            
        Returns:
            List of ranked sections with scores
        """
        try:
            self.logger.info(f"Ranking sections for query: {query_context}")
            
            # Collect all sections from all documents
            all_sections = []
            for filename, parsed_doc in parsed_documents.items():
                if "sections" in parsed_doc:
                    all_sections.extend(parsed_doc["sections"])
            
            if not all_sections:
                self.logger.warning("No sections found in any document")
                return []
            
            self.logger.info(f"Found {len(all_sections)} total sections to rank")
            
            # Filter out very short sections (likely noise)
            filtered_sections = [
                section for section in all_sections 
                if section.get("word_count", 0) >= 10  # Minimum 10 words
            ]
            
            self.logger.info(f"After filtering: {len(filtered_sections)} sections")
            
            if not filtered_sections:
                return []
            
            # Limit to max_sections to avoid memory/time issues
            if len(filtered_sections) > max_sections:
                # Sort by word count and take the longer sections
                filtered_sections.sort(key=lambda x: x.get("word_count", 0), reverse=True)
                filtered_sections = filtered_sections[:max_sections]
                self.logger.info(f"Limited to {max_sections} longest sections")
            
            # Score each section
            scored_sections = []
            for section in filtered_sections:
                score = self._score_section(section, query_context)
                
                scored_section = {
                    "title": section["title"],
                    "content": section["content"],
                    "page_number": section["page_number"],
                    "document": section["document"],
                    "word_count": section.get("word_count", 0),
                    "relevance_score": score
                }
                scored_sections.append(scored_section)
            
            # Sort by relevance score (descending)
            scored_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            top_scores = [f"{s['relevance_score']:.3f}" for s in scored_sections[:5]]
            self.logger.info(f"Top 5 section scores: {top_scores}")
            
            return scored_sections
            
        except Exception as e:
            self.logger.error(f"Error ranking sections: {str(e)}")
            return []
    
    def _score_section(self, section: Dict, query_context: str) -> float:
        """
        Score a single section based on relevance to query context
        
        Args:
            section: Section dictionary with title and content
            query_context: Combined persona and job-to-be-done context
            
        Returns:
            Relevance score (0-1)
        """
        try:
            # Combine title and content for scoring
            section_text = f"{section['title']} {section['content']}"
            
            # Get embeddings
            query_embedding = self.vectorizer.encode_single_text(query_context)
            section_embedding = self.vectorizer.encode_single_text(section_text)
            
            # Compute base similarity
            base_similarity = self.vectorizer.compute_similarity(
                query_embedding, section_embedding
            )
            
            # Apply additional scoring factors
            title_bonus = self._score_title_relevance(section["title"], query_context)
            content_bonus = self._score_content_quality(section)
            length_bonus = self._score_section_length(section.get("word_count", 0))
            
            # Combine scores with weights
            final_score = (
                base_similarity * 0.70 +  # Primary semantic similarity
                title_bonus * 0.15 +     # Title relevance bonus
                content_bonus * 0.10 +   # Content quality bonus  
                length_bonus * 0.05      # Length appropriateness bonus
            )
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error scoring section: {str(e)}")
            return 0.0
    
    def _score_title_relevance(self, title: str, query_context: str) -> float:
        """
        Score how relevant the section title is to the query
        
        Args:
            title: Section title
            query_context: Query context
            
        Returns:
            Title relevance score (0-1)
        """
        try:
            if not title or not query_context:
                return 0.0
            
            # Direct keyword matching bonus
            title_lower = title.lower()
            query_lower = query_context.lower()
            
            # Extract keywords from query
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            
            # Calculate word overlap
            common_words = query_words.intersection(title_words)
            if query_words:
                word_overlap = len(common_words) / len(query_words)
            else:
                word_overlap = 0.0
            
            # Semantic similarity between title and query
            title_embedding = self.vectorizer.encode_single_text(title)
            query_embedding = self.vectorizer.encode_single_text(query_context)
            semantic_sim = self.vectorizer.compute_similarity(title_embedding, query_embedding)
            
            # Combine with weights
            title_score = word_overlap * 0.4 + semantic_sim * 0.6
            
            return min(1.0, title_score)
            
        except Exception as e:
            self.logger.error(f"Error scoring title relevance: {str(e)}")
            return 0.0
    
    def _score_content_quality(self, section: Dict) -> float:
        """
        Score the quality/richness of section content
        
        Args:
            section: Section dictionary
            
        Returns:
            Content quality score (0-1)
        """
        try:
            content = section.get("content", "")
            if not content:
                return 0.0
            
            # Factors that indicate good content
            sentences = content.split('.')
            num_sentences = len([s for s in sentences if s.strip()])
            
            # Prefer sections with multiple sentences
            sentence_score = min(1.0, num_sentences / 5.0)  # Normalize to 5 sentences
            
            # Check for structured content (lists, numbered items)
            has_structure = any(marker in content for marker in [':', '-', '•', '1.', '2.', 'a)', 'b)'])
            structure_bonus = 0.2 if has_structure else 0.0
            
            # Check for specific information (numbers, dates, names)
            import re
            has_specifics = bool(re.search(r'\d+|[A-Z][a-z]+\s[A-Z][a-z]+', content))
            specifics_bonus = 0.1 if has_specifics else 0.0
            
            quality_score = sentence_score + structure_bonus + specifics_bonus
            
            return min(1.0, quality_score)
            
        except Exception as e:
            self.logger.error(f"Error scoring content quality: {str(e)}")
            return 0.0
    
    def _score_section_length(self, word_count: int) -> float:
        """
        Score section based on appropriate length (not too short or too long)
        
        Args:
            word_count: Number of words in section
            
        Returns:
            Length score (0-1)
        """
        try:
            if word_count <= 0:
                return 0.0
            
            # Prefer sections with 50-300 words (good balance)
            if 50 <= word_count <= 300:
                return 1.0
            elif 20 <= word_count < 50:
                return 0.8  # Slightly shorter is okay
            elif 300 < word_count <= 500:
                return 0.7  # Slightly longer is okay
            elif 10 <= word_count < 20:
                return 0.4  # Too short
            elif 500 < word_count <= 1000:
                return 0.5  # Too long
            else:
                return 0.2  # Very short or very long
            
        except Exception as e:
            self.logger.error(f"Error scoring section length: {str(e)}")
            return 0.0
    
    def get_ranking_stats(self, ranked_sections: List[Dict]) -> Dict:
        """
        Get statistics about the ranking results
        
        Args:
            ranked_sections: List of ranked sections
            
        Returns:
            Dictionary with ranking statistics
        """
        if not ranked_sections:
            return {
                "total_sections": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "score_distribution": []
            }
        
        scores = [section["relevance_score"] for section in ranked_sections]
        
        return {
            "total_sections": len(ranked_sections),
            "avg_score": np.mean(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "score_distribution": {
                "high (>0.7)": len([s for s in scores if s > 0.7]),
                "medium (0.4-0.7)": len([s for s in scores if 0.4 <= s <= 0.7]),
                "low (<0.4)": len([s for s in scores if s < 0.4])
            }
        } 