"""
PDF Parser component for extracting structured text from PDF documents
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    import PyMuPDF as fitz

class PDFParser:
    """Parses PDF documents and extracts structured text with metadata"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse_pdf(self, pdf_path: Path) -> Dict:
        """
        Parse a PDF file and extract structured content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing parsed document structure
        """
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            
            # Store document info before processing
            total_pages = len(doc)
            
            parsed_doc = {
                "filename": pdf_path.name,
                "total_pages": total_pages,
                "sections": [],
                "raw_text": ""
            }
            
            full_text = ""
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract sections from this page
                sections = self._extract_sections_from_page(
                    page_text, 
                    page_num + 1, 
                    pdf_path.name
                )
                parsed_doc["sections"].extend(sections)
            
            parsed_doc["raw_text"] = full_text
            
            # Close document after processing
            doc.close()
            doc = None  # Set to None to prevent accidental access
            
            self.logger.info(f"Parsed {pdf_path.name}: {total_pages} pages, {len(parsed_doc['sections'])} sections")
            
            return parsed_doc
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
            return {
                "filename": pdf_path.name,
                "total_pages": 0,
                "sections": [],
                "raw_text": "",
                "error": str(e)
            }
    
    def _extract_sections_from_page(self, page_text: str, page_num: int, filename: str) -> List[Dict]:
        """
        Extract sections from a single page of text
        
        Args:
            page_text: Raw text from the page
            page_num: Page number (1-indexed)
            filename: Name of the PDF file
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        
        current_section = None
        section_content = []
        
        for paragraph in paragraphs:
            # Check if this paragraph looks like a heading/title
            if self._is_heading(paragraph):
                # Save previous section if exists
                if current_section and section_content:
                    sections.append({
                        "title": current_section,
                        "content": "\n\n".join(section_content),
                        "page_number": page_num,
                        "document": filename,
                        "word_count": len(" ".join(section_content).split())
                    })
                
                # Start new section
                current_section = paragraph
                section_content = []
            else:
                # Add to current section content
                if paragraph:
                    section_content.append(paragraph)
        
        # Add final section if exists
        if current_section and section_content:
            sections.append({
                "title": current_section,
                "content": "\n\n".join(section_content),
                "page_number": page_num,
                "document": filename,
                "word_count": len(" ".join(section_content).split())
            })
        
        # If no clear sections found, create one section for the entire page
        if not sections and page_text.strip():
            # Try to extract a title from the first few lines
            lines = page_text.strip().split('\n')
            title = self._extract_page_title(lines)
            
            sections.append({
                "title": title,
                "content": page_text.strip(),
                "page_number": page_num,
                "document": filename,
                "word_count": len(page_text.split())
            })
        
        return sections
    
    def _is_heading(self, text: str) -> bool:
        """
        Determine if a text block is likely a heading/title
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if text is likely a heading
        """
        if not text or len(text) > 200:  # Too long to be a heading
            return False
        
        # Check for heading patterns
        heading_patterns = [
            r'^[A-Z][A-Za-z\s\-:]+$',  # Title case
            r'^[A-Z\s\-:]+$',          # All caps
            r'^\d+\.?\s+[A-Za-z]',     # Numbered heading
            r'^[IVX]+\.?\s+[A-Za-z]',  # Roman numeral heading
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # Check structural indicators
        lines = text.split('\n')
        if len(lines) == 1:  # Single line
            words = text.split()
            if 2 <= len(words) <= 10:  # Reasonable title length
                return True
        
        return False
    
    def _extract_page_title(self, lines: List[str]) -> str:
        """
        Extract a title from the first few lines of a page
        
        Args:
            lines: List of text lines from the page
            
        Returns:
            Extracted title or default title
        """
        # Look for title in first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if line and len(line.split()) >= 2:
                # Check if it looks like a title
                if (line[0].isupper() or 
                    any(word[0].isupper() for word in line.split()) or
                    self._is_heading(line)):
                    return line
        
        # Fallback: use first non-empty line or generate generic title
        for line in lines[:3]:
            if line.strip():
                return line.strip()[:100]  # Limit length
        
        return "Document Content"
    
    def get_document_stats(self, parsed_doc: Dict) -> Dict:
        """
        Get statistics about the parsed document
        
        Args:
            parsed_doc: Parsed document structure
            
        Returns:
            Dictionary with document statistics
        """
        total_words = sum(section.get("word_count", 0) for section in parsed_doc["sections"])
        
        return {
            "filename": parsed_doc["filename"],
            "total_pages": parsed_doc["total_pages"],
            "total_sections": len(parsed_doc["sections"]),
            "total_words": total_words,
            "avg_words_per_section": total_words / len(parsed_doc["sections"]) if parsed_doc["sections"] else 0
        } 