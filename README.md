# Adobe India Hackathon Round 1B - PDF Document Persona System

This is a complete solution for Round 1B of the Adobe India Hackathon that extracts and ranks the most relevant sections from a collection of PDFs based on a given persona and job-to-be-done.

## Overview

The system processes 3-10 PDF documents and identifies the most relevant sections and subsections that would help a specific persona achieve their stated goal. It uses advanced NLP techniques including semantic similarity, content analysis, and extractive summarization.

## Architecture

### Core Components

1. **PDF Parser** (`parser.py`)
   - Extracts structured text from PDFs using PyMuPDF
   - Identifies sections and headings
   - Preserves page number information

2. **Document Vectorizer** (`vectorizer.py`)
   - Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
   - Computes cosine similarity between content and queries
   - Optimized for CPU-only execution

3. **Relevance Ranker** (`ranker.py`)
   - Multi-factor scoring system combining:
     - Semantic similarity (70%)
     - Title relevance (15%)
     - Content quality (10%)
     - Section length appropriateness (5%)

4. **Content Summarizer** (`summarizer.py`)
   - Extractive summarization focusing on query relevance
   - Sentence-level scoring and selection
   - Length-aware content refinement

5. **Schema Validator** (`schema.py`)
   - Ensures output format compliance
   - Validates input configuration structure
   - Consistency checking between components

6. **Utilities** (`utils.py`)
   - Configuration loading and validation
   - Logging setup and performance monitoring
   - Text processing helpers

## Input Format

The system expects:

1. **PDF Documents**: 3-10 PDF files in `/app/input/` directory
2. **Configuration File**: `challenge1b_input.json` with structure:

```json
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document 1 Title"
    }
  ]
}
```

## Output Format

Generates `output.json` with:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip...",
    "processing_timestamp": "2025-01-28T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Section Title",
      "importance_rank": 1,
      "page_number": 4
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Refined summary text...",
      "page_number": 4
    }
  ]
}
```

## Technical Specifications

### Performance Constraints
- **Runtime**: ≤ 60 seconds for 3-5 documents
- **Model Size**: ≤ 1GB total (uses ~80MB MiniLM model)
- **Memory**: Optimized for 16GB RAM limit
- **CPU**: AMD64 architecture, no GPU required

### Dependencies
- **Python**: 3.10
- **Core Libraries**:
  - PyMuPDF for PDF processing
  - sentence-transformers for embeddings
  - torch (CPU-only)
  - numpy, scikit-learn for numerical operations

## Build and Run Instructions

### Docker Build

```bash
docker build --platform linux/amd64 -t adobe-insight .
```

### Docker Run

```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none adobe-insight
```

### Local Development Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run locally
cd app
python main.py
```

## Testing with Sample Data

```bash
# Copy sample data
cp -r sample/Collection\ 1/* input/

# Run the system
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none adobe-insight

# Check output
cat output/output.json
```

## Algorithm Details

### Relevance Scoring

The system uses a multi-factor scoring approach:

1. **Semantic Similarity** (70% weight)
   - Embeds persona+job and section content
   - Computes cosine similarity
   - Primary ranking factor

2. **Title Relevance** (15% weight)
   - Keyword overlap between title and query
   - Semantic similarity of title to query
   - Bonus for relevant section headings

3. **Content Quality** (10% weight)
   - Sentence count and structure
   - Presence of lists, numbers, specific information
   - Content richness indicators

4. **Length Appropriateness** (5% weight)
   - Optimal range: 50-300 words
   - Penalizes very short or very long sections
   - Ensures meaningful content

### Summarization Strategy

1. **Sentence Extraction**: Split content into sentences
2. **Relevance Scoring**: Score each sentence against query context
3. **Selection**: Choose top sentences up to length limit
4. **Cleaning**: Remove incomplete sentences, normalize format

## Performance Optimizations

- **Model Caching**: Pre-downloads model during Docker build
- **Batch Processing**: Efficient embedding generation
- **Memory Management**: Limits section processing to avoid OOM
- **CPU Optimization**: Uses torch CPU tensors exclusively

## Error Handling

- Graceful PDF parsing failures
- Fallback text extraction methods
- Validation of all input/output formats
- Comprehensive logging for debugging

## Multilingual Support

- Uses multilingual sentence transformer model
- Unicode-aware text processing
- Language-agnostic ranking features

## Limitations

- Text-based PDFs only (no OCR for scanned documents)
- Maximum 50 sections processed per run (performance limit)
- English-optimized but supports other languages
- CPU-only execution (no GPU acceleration)
