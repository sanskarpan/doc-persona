#!/usr/bin/env python3
"""
Adobe India Hackathon Round 1B - PDF Document Persona System
Main entry point that processes PDF collections and extracts relevant sections
based on persona and job-to-be-done requirements.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import time

from parser import PDFParser
from vectorizer import DocumentVectorizer
from ranker import RelevanceRanker
from summarizer import ContentSummarizer
from schema import OutputValidator
from utils import setup_logging, load_input_config

def main():
    """Main processing pipeline for PDF document analysis"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define input/output paths (Docker or local)
    if Path("/app/input").exists():
        # Running in Docker
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
    else:
        # Running locally
        input_dir = Path("../input")
        output_dir = Path("../output")
    
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Load input configuration
        logger.info("Loading input configuration...")
        config = load_input_config(input_dir)
        
        if not config:
            logger.error("No valid input configuration found")
            sys.exit(1)
            
        persona = config.get("persona", {}).get("role", "")
        job_to_be_done = config.get("job_to_be_done", {}).get("task", "")
        documents = config.get("documents", [])
        
        logger.info(f"Processing {len(documents)} documents for persona: {persona}")
        logger.info(f"Job to be done: {job_to_be_done}")
        
        # Initialize components
        pdf_parser = PDFParser()
        vectorizer = DocumentVectorizer()
        ranker = RelevanceRanker(vectorizer)
        summarizer = ContentSummarizer()
        validator = OutputValidator()
        
        # Parse all PDFs
        logger.info("Parsing PDF documents...")
        parsed_documents = {}
        for doc_info in documents:
            filename = doc_info["filename"]
            pdf_path = input_dir / filename
            
            if pdf_path.exists():
                logger.info(f"Parsing {filename}...")
                parsed_doc = pdf_parser.parse_pdf(pdf_path)
                parsed_documents[filename] = parsed_doc
            else:
                logger.warning(f"PDF file not found: {filename}")
        
        if not parsed_documents:
            logger.error("No PDF documents were successfully parsed")
            sys.exit(1)
        
        # Create query context from persona and job
        query_context = f"{persona}: {job_to_be_done}"
        
        # Rank sections based on relevance
        logger.info("Ranking sections by relevance...")
        ranked_sections = ranker.rank_sections(parsed_documents, query_context)
        
        # Select top sections (limit to top 5 for output)
        top_sections = ranked_sections[:5]
        
        # Generate subsection analysis
        logger.info("Generating subsection analysis...")
        subsection_analysis = []
        for section in top_sections:
            refined_text = summarizer.summarize_section(
                section["content"], 
                query_context,
                max_length=500
            )
            
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"]
            })
        
        # Build output structure
        output_data = {
            "metadata": {
                "input_documents": [doc["filename"] for doc in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section["document"],
                    "section_title": section["title"],
                    "importance_rank": idx + 1,
                    "page_number": section["page_number"]
                }
                for idx, section in enumerate(top_sections)
            ],
            "subsection_analysis": subsection_analysis
        }
        
        # Validate output
        if not validator.validate_output(output_data):
            logger.error("Output validation failed")
            sys.exit(1)
        
        # Save output
        output_file = output_dir / "output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Extracted {len(top_sections)} relevant sections")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 