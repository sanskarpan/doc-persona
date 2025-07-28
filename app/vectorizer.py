"""
Document Vectorizer component for creating embeddings and computing similarity
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import torch

class DocumentVectorizer:
    """Handles document vectorization and similarity computation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vectorizer with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            # Load the sentence transformer model
            self.logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Set to CPU mode for consistency
            if torch.cuda.is_available():
                self.logger.info("GPU available but using CPU for consistency")
            
            device = torch.device('cpu')
            self.model = self.model.to(device)
            
            self.logger.info(f"Model loaded successfully on device: {device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings
        """
        try:
            if not texts:
                return np.array([])
            
            # Clean and prepare texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {str(e)}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 384))  # MiniLM-L6 embedding size
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding
        
        Args:
            text: Text string to encode
            
        Returns:
            NumPy array embedding
        """
        return self.encode_texts([text])[0]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings (N x D)
            embeddings2: Second set of embeddings (M x D)
            
        Returns:
            Similarity matrix (N x M)
        """
        try:
            if embeddings1.size == 0 or embeddings2.size == 0:
                return np.array([])
            
            # Normalize embeddings
            norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norms1 = np.where(norms1 == 0, 1, norms1)
            norms2 = np.where(norms2 == 0, 1, norms2)
            
            normalized1 = embeddings1 / norms1
            normalized2 = embeddings2 / norms2
            
            # Compute cosine similarity matrix
            similarity_matrix = np.dot(normalized1, normalized2.T)
            
            # Convert to [0, 1] range
            similarity_matrix = (similarity_matrix + 1) / 2
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing similarity matrix: {str(e)}")
            return np.zeros((embeddings1.shape[0], embeddings2.shape[0]))
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embeddings matrix
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        try:
            if candidate_embeddings.size == 0:
                return []
            
            # Compute similarities
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding most similar: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for encoding
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Limit length to prevent issues with very long texts
        max_length = 512  # Conservative limit for sentence transformers
        if len(cleaned) > max_length:
            # Try to cut at sentence boundary
            sentences = cleaned.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) > max_length:
                    break
                truncated += sentence + ". "
            cleaned = truncated.strip()
            
            # If still too long, hard truncate
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length]
        
        return cleaned
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "max_seq_length": getattr(self.model, 'max_seq_length', 512),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": str(self.model.device)
        } 