"""
Schema validation component for output format compliance
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

class OutputValidator:
    """Validates output JSON structure and content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_output(self, output_data: Dict) -> bool:
        """
        Validate the complete output structure
        
        Args:
            output_data: Output dictionary to validate
            
        Returns:
            Boolean indicating if output is valid
        """
        try:
            # Check top-level structure
            required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
            for key in required_keys:
                if key not in output_data:
                    self.logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate metadata
            if not self._validate_metadata(output_data["metadata"]):
                return False
            
            # Validate extracted sections
            if not self._validate_extracted_sections(output_data["extracted_sections"]):
                return False
            
            # Validate subsection analysis
            if not self._validate_subsection_analysis(output_data["subsection_analysis"]):
                return False
            
            # Check consistency between sections
            if not self._validate_consistency(output_data):
                return False
            
            self.logger.info("Output validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False
    
    def _validate_metadata(self, metadata: Dict) -> bool:
        """
        Validate metadata structure
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Boolean indicating if metadata is valid
        """
        required_fields = [
            "input_documents",
            "persona", 
            "job_to_be_done",
            "processing_timestamp"
        ]
        
        for field in required_fields:
            if field not in metadata:
                self.logger.error(f"Missing metadata field: {field}")
                return False
        
        # Validate input_documents is a list
        if not isinstance(metadata["input_documents"], list):
            self.logger.error("input_documents must be a list")
            return False
        
        if len(metadata["input_documents"]) == 0:
            self.logger.error("input_documents cannot be empty")
            return False
        
        # Validate persona and job_to_be_done are strings
        if not isinstance(metadata["persona"], str) or not metadata["persona"].strip():
            self.logger.error("persona must be a non-empty string")
            return False
        
        if not isinstance(metadata["job_to_be_done"], str) or not metadata["job_to_be_done"].strip():
            self.logger.error("job_to_be_done must be a non-empty string")
            return False
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(metadata["processing_timestamp"].replace('Z', '+00:00'))
        except ValueError:
            self.logger.error("Invalid timestamp format")
            return False
        
        return True
    
    def _validate_extracted_sections(self, sections: List[Dict]) -> bool:
        """
        Validate extracted sections structure
        
        Args:
            sections: List of section dictionaries
            
        Returns:
            Boolean indicating if sections are valid
        """
        if not isinstance(sections, list):
            self.logger.error("extracted_sections must be a list")
            return False
        
        if len(sections) == 0:
            self.logger.error("extracted_sections cannot be empty")
            return False
        
        required_fields = ["document", "section_title", "importance_rank", "page_number"]
        
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                self.logger.error(f"Section {i} must be a dictionary")
                return False
            
            for field in required_fields:
                if field not in section:
                    self.logger.error(f"Section {i} missing field: {field}")
                    return False
            
            # Validate field types and values
            if not isinstance(section["document"], str) or not section["document"].strip():
                self.logger.error(f"Section {i} document must be a non-empty string")
                return False
            
            if not isinstance(section["section_title"], str) or not section["section_title"].strip():
                self.logger.error(f"Section {i} section_title must be a non-empty string")
                return False
            
            if not isinstance(section["importance_rank"], int) or section["importance_rank"] < 1:
                self.logger.error(f"Section {i} importance_rank must be a positive integer")
                return False
            
            if not isinstance(section["page_number"], int) or section["page_number"] < 1:
                self.logger.error(f"Section {i} page_number must be a positive integer")
                return False
        
        # Check that importance ranks are consecutive starting from 1
        ranks = [section["importance_rank"] for section in sections]
        expected_ranks = list(range(1, len(sections) + 1))
        if sorted(ranks) != expected_ranks:
            self.logger.error("Importance ranks must be consecutive starting from 1")
            return False
        
        return True
    
    def _validate_subsection_analysis(self, analysis: List[Dict]) -> bool:
        """
        Validate subsection analysis structure
        
        Args:
            analysis: List of analysis dictionaries
            
        Returns:
            Boolean indicating if analysis is valid
        """
        if not isinstance(analysis, list):
            self.logger.error("subsection_analysis must be a list")
            return False
        
        required_fields = ["document", "refined_text", "page_number"]
        
        for i, item in enumerate(analysis):
            if not isinstance(item, dict):
                self.logger.error(f"Analysis item {i} must be a dictionary")
                return False
            
            for field in required_fields:
                if field not in item:
                    self.logger.error(f"Analysis item {i} missing field: {field}")
                    return False
            
            # Validate field types and values
            if not isinstance(item["document"], str) or not item["document"].strip():
                self.logger.error(f"Analysis item {i} document must be a non-empty string")
                return False
            
            if not isinstance(item["refined_text"], str) or not item["refined_text"].strip():
                self.logger.error(f"Analysis item {i} refined_text must be a non-empty string")
                return False
            
            if not isinstance(item["page_number"], int) or item["page_number"] < 1:
                self.logger.error(f"Analysis item {i} page_number must be a positive integer")
                return False
        
        return True
    
    def _validate_consistency(self, output_data: Dict) -> bool:
        """
        Validate consistency between different sections of output
        
        Args:
            output_data: Complete output data
            
        Returns:
            Boolean indicating if output is consistent
        """
        metadata = output_data["metadata"]
        extracted_sections = output_data["extracted_sections"]
        subsection_analysis = output_data["subsection_analysis"]
        
        # Check that all documents in sections are listed in metadata
        input_docs = set(metadata["input_documents"])
        section_docs = set(section["document"] for section in extracted_sections)
        analysis_docs = set(item["document"] for item in subsection_analysis)
        
        if not section_docs.issubset(input_docs):
            self.logger.error("Some section documents not listed in input_documents")
            return False
        
        if not analysis_docs.issubset(input_docs):
            self.logger.error("Some analysis documents not listed in input_documents")
            return False
        
        # Check that the number of extracted sections matches analysis items
        # (This might not always be true, but it's expected in most cases)
        if len(extracted_sections) != len(subsection_analysis):
            self.logger.warning(
                f"Number of extracted sections ({len(extracted_sections)}) "
                f"differs from analysis items ({len(subsection_analysis)})"
            )
        
        return True
    
    def validate_input_config(self, config: Dict) -> bool:
        """
        Validate input configuration structure
        
        Args:
            config: Input configuration dictionary
            
        Returns:
            Boolean indicating if config is valid
        """
        try:
            # Check required top-level fields
            if "persona" not in config:
                self.logger.error("Missing persona in input config")
                return False
            
            if "job_to_be_done" not in config:
                self.logger.error("Missing job_to_be_done in input config")
                return False
            
            if "documents" not in config:
                self.logger.error("Missing documents in input config")
                return False
            
            # Validate persona structure
            persona = config["persona"]
            if not isinstance(persona, dict) or "role" not in persona:
                self.logger.error("Invalid persona structure")
                return False
            
            # Validate job_to_be_done structure
            job = config["job_to_be_done"]
            if not isinstance(job, dict) or "task" not in job:
                self.logger.error("Invalid job_to_be_done structure")
                return False
            
            # Validate documents structure
            documents = config["documents"]
            if not isinstance(documents, list) or len(documents) == 0:
                self.logger.error("Documents must be a non-empty list")
                return False
            
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    self.logger.error(f"Document {i} must be a dictionary")
                    return False
                
                if "filename" not in doc:
                    self.logger.error(f"Document {i} missing filename")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input config: {str(e)}")
            return False
    
    def get_validation_summary(self, output_data: Dict) -> Dict:
        """
        Get summary of validation results
        
        Args:
            output_data: Output data to analyze
            
        Returns:
            Validation summary dictionary
        """
        summary = {
            "valid": False,
            "total_sections": 0,
            "total_analysis_items": 0,
            "total_documents": 0,
            "documents_with_sections": 0,
            "issues": []
        }
        
        try:
            if self.validate_output(output_data):
                summary["valid"] = True
            
            # Count items
            summary["total_sections"] = len(output_data.get("extracted_sections", []))
            summary["total_analysis_items"] = len(output_data.get("subsection_analysis", []))
            summary["total_documents"] = len(output_data.get("metadata", {}).get("input_documents", []))
            
            # Count documents with sections
            if "extracted_sections" in output_data:
                unique_docs = set(section["document"] for section in output_data["extracted_sections"])
                summary["documents_with_sections"] = len(unique_docs)
            
        except Exception as e:
            summary["issues"].append(f"Error during validation: {str(e)}")
        
        return summary 