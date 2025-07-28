"""
Utility functions for the PDF Document Persona System
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up logging to console
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def load_input_config(input_dir: Path) -> Optional[Dict]:
    """
    Load input configuration from input directory
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        Configuration dictionary or None if not found
    """
    logger = logging.getLogger(__name__)
    
    # Look for different possible input file names
    possible_files = [
        "challenge1b_input.json",
        "persona.json", 
        "input.json",
        "config.json"
    ]
    
    for filename in possible_files:
        config_path = input_dir / filename
        if config_path.exists():
            try:
                logger.info(f"Loading configuration from: {filename}")
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Validate basic structure
                if _validate_basic_config(config):
                    return config
                else:
                    logger.warning(f"Invalid configuration in {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                continue
    
    logger.error("No valid input configuration found")
    return None

def _validate_basic_config(config: Dict) -> bool:
    """
    Validate basic configuration structure
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Boolean indicating if config is valid
    """
    # Check for required fields
    required_fields = ["persona", "job_to_be_done", "documents"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    # Check persona structure
    if not isinstance(config["persona"], dict) or "role" not in config["persona"]:
        return False
    
    # Check job_to_be_done structure  
    if not isinstance(config["job_to_be_done"], dict) or "task" not in config["job_to_be_done"]:
        return False
    
    # Check documents structure
    if not isinstance(config["documents"], list) or len(config["documents"]) == 0:
        return False
    
    return True

def get_pdf_files(input_dir: Path) -> List[Path]:
    """
    Get list of PDF files in input directory
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        List of PDF file paths
    """
    try:
        pdf_files = list(input_dir.glob("*.pdf"))
        pdf_files.extend(input_dir.glob("**/*.pdf"))  # Include subdirectories
        return sorted(pdf_files)
    except Exception:
        return []

def safe_filename(filename: str) -> str:
    """
    Convert filename to safe format for file system
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe = re.sub(r'\s+', '_', safe)
    return safe.strip('_')

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def calculate_processing_stats(start_time: float, end_time: float, 
                              num_documents: int, num_sections: int) -> Dict:
    """
    Calculate processing statistics
    
    Args:
        start_time: Processing start time
        end_time: Processing end time
        num_documents: Number of documents processed
        num_sections: Number of sections processed
        
    Returns:
        Statistics dictionary
    """
    processing_time = end_time - start_time
    
    return {
        "processing_time_seconds": round(processing_time, 2),
        "documents_processed": num_documents,
        "sections_processed": num_sections,
        "avg_time_per_document": round(processing_time / num_documents, 2) if num_documents > 0 else 0,
        "avg_time_per_section": round(processing_time / num_sections, 2) if num_sections > 0 else 0,
        "sections_per_second": round(num_sections / processing_time, 2) if processing_time > 0 else 0
    }

def clean_text_for_json(text: str) -> str:
    """
    Clean text for safe JSON serialization
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove control characters that can break JSON
    import re
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()

def ensure_directory_exists(directory: Path) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Path to directory
        
    Returns:
        Boolean indicating success
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def get_system_info() -> Dict:
    """
    Get system information for debugging
    
    Returns:
        System information dictionary
    """
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        }
    except Exception:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": "unknown",
            "memory_gb": "unknown",
            "available_memory_gb": "unknown"
        }

def log_performance_metrics(logger: logging.Logger, metrics: Dict) -> None:
    """
    Log performance metrics in a structured way
    
    Args:
        logger: Logger instance
        metrics: Metrics dictionary
    """
    logger.info("=== Performance Metrics ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("==========================")

def validate_file_exists(file_path: Path, logger: logging.Logger) -> bool:
    """
    Validate that a file exists and log appropriate message
    
    Args:
        file_path: Path to file
        logger: Logger instance
        
    Returns:
        Boolean indicating if file exists
    """
    if file_path.exists():
        logger.debug(f"File exists: {file_path}")
        return True
    else:
        logger.warning(f"File not found: {file_path}")
        return False 