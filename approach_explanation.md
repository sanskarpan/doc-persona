# Approach Explanation: PDF Document Persona System

## Architecture Overview

Our solution implements a sophisticated multi-stage pipeline that combines traditional NLP techniques with modern transformer-based embeddings to accurately match document content with user personas and objectives.

## Core Design Decisions

### 1. Semantic Similarity as Primary Ranking Factor

We chose to make semantic similarity the dominant factor (70% weight) in our scoring algorithm because it most accurately captures the intent behind a persona's job-to-be-done. By encoding both the query context ("Travel Planner: Plan a trip of 4 days for a group of 10 college friends") and document sections using the same embedding space, we can identify conceptually relevant content even when exact keyword matches are absent.

The all-MiniLM-L6-v2 model was selected for its optimal balance of accuracy, size (<100MB), and inference speed. This multilingual model ensures broad language support while meeting strict resource constraints.

### 2. Multi-Factor Scoring System

Rather than relying solely on semantic similarity, we implemented a weighted combination of four factors:

- **Semantic Similarity (70%)**: Primary relevance measure using cosine similarity between embeddings
- **Title Relevance (15%)**: Specific attention to section headings, which often contain the most concentrated topic information
- **Content Quality (10%)**: Rewards structured, information-rich content with specific details, numbers, and organized formatting
- **Length Appropriateness (5%)**: Ensures optimal section length (50-300 words) for meaningful analysis

This approach prevents the system from favoring either extremely verbose or overly brief sections while prioritizing semantic relevance.

### 3. Extractive Summarization Strategy

We chose extractive over abstractive summarization for several reasons:
- **Consistency**: Preserves original document language and terminology
- **Reliability**: Avoids hallucination risks inherent in generative approaches  
- **Performance**: Faster execution within 60-second constraint
- **Accuracy**: Maintains factual integrity of source material

Our sentence-level scoring approach identifies the most query-relevant sentences while maintaining coherent, complete thoughts in the output.

### 4. Hierarchical Text Processing

The system processes documents at multiple granularities:
1. **Document Level**: Overall PDF structure and metadata
2. **Section Level**: Identified through heading detection and paragraph clustering
3. **Sentence Level**: For fine-grained relevance scoring during summarization

This hierarchical approach allows for both broad content categorization and precise information extraction.

## Technical Implementation Highlights

### PDF Processing Pipeline
We implemented robust PDF parsing using PyMuPDF with fallback mechanisms for various document formats. The system automatically detects section boundaries using pattern matching for common heading formats while gracefully handling unstructured documents.

### Memory and Performance Optimization
- **Batch Processing**: Efficient embedding generation for multiple text segments
- **Section Limiting**: Caps processing at 50 sections to prevent memory overflow
- **Model Caching**: Pre-loads transformer model during Docker build to reduce runtime
- **CPU Optimization**: Specifically tuned for CPU-only execution environments

### Error Resilience
The system includes comprehensive error handling with graceful degradation:
- PDF parsing failures fall back to alternative extraction methods
- Missing or malformed input files generate appropriate warnings
- Invalid output structures trigger validation errors with specific feedback

## Scoring Logic Rationale

Our scoring algorithm balances multiple signals to identify truly relevant content:

1. **Primary Relevance**: Semantic embeddings capture deep conceptual relationships between user intent and document content
2. **Surface Signals**: Title analysis provides quick relevance indicators for well-structured documents  
3. **Quality Metrics**: Content analysis ensures selected sections contain substantial, useful information
4. **Practical Constraints**: Length scoring optimizes for human-readable, actionable content segments

The weighted combination ensures that high-quality, relevant content rises to the top while filtering out noise from poorly structured or tangentially related sections.

## Design Trade-offs

### Accuracy vs. Speed
We prioritized accuracy over maximum speed by implementing thorough content analysis rather than simple keyword matching. The 60-second runtime constraint is met through efficient algorithms rather than sacrificing analysis depth.

### Simplicity vs. Sophistication  
While we could have implemented more complex neural architectures, the current approach balances sophistication with reliability. The modular design allows for easy component replacement or enhancement.

### Generalizability vs. Optimization
The system is designed to work across diverse document types and personas rather than being optimized for specific domains. This ensures robust performance across the varied test cases in the hackathon evaluation.

This approach delivers a production-ready system that consistently identifies the most relevant content for any given persona and objective while meeting all technical constraints and performance requirements. 