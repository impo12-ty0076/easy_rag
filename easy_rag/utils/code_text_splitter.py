"""
Code text splitter utilities for Easy RAG System.
This module provides functionality for splitting code into chunks for vector databases.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from easy_rag.utils.text_splitters import TextSplitter

class CodeTextSplitter(TextSplitter):
    """
    Code Text Splitter that splits code using language-specific separators.
    Uses langchain_text_splitters.RecursiveCharacterTextSplitter with language-specific settings.
    
    코드 텍스트 분할기는 언어별 구분자를 사용하여 코드를 분할합니다.
    langchain_text_splitters.RecursiveCharacterTextSplitter를 언어별 설정과 함께 사용합니다.
    """
    
    name = "Code Text Splitter"
    description = "Splits code based on language-specific syntax"
    
    # Map of supported languages
    SUPPORTED_LANGUAGES = {
        lang.name: lang for lang in Language
    }
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for code text splitter"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'language': 'PYTHON',
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           language: str = 'PYTHON') -> Tuple[bool, str]:
        """Validate parameters for code text splitter"""
        if not (100 <= chunk_size <= 8000):
            return False, "Chunk size must be between 100 and 8000 characters"
        
        if not (0 <= chunk_overlap <= 500):
            return False, "Chunk overlap must be between 0 and 500 characters"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if language not in cls.SUPPORTED_LANGUAGES:
            return False, f"Language must be one of: {', '.join(cls.SUPPORTED_LANGUAGES.keys())}"
        
        return True, ""
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported programming languages"""
        return list(cls.SUPPORTED_LANGUAGES.keys())
    
    @classmethod
    def get_separators_for_language(cls, language: str) -> List[str]:
        """Get the separators used for a specific language"""
        if language not in cls.SUPPORTED_LANGUAGES:
            return []
        
        lang_enum = cls.SUPPORTED_LANGUAGES[language]
        return RecursiveCharacterTextSplitter.get_separators_for_language(lang_enum)
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int, 
                  language: str = 'PYTHON') -> List[str]:
        """
        Split code into chunks based on language-specific syntax
        
        Args:
            text: The code text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            language: Programming language of the code
            
        Returns:
            List of code chunks
        """
        if not text:
            return []
        
        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            # Default to Python if language not supported
            language = 'PYTHON'
        
        # Get the language enum
        lang_enum = self.SUPPORTED_LANGUAGES[language]
        
        # Create a RecursiveCharacterTextSplitter with language-specific settings
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split the code
        docs = splitter.create_documents([text])
        
        # Extract the content from the documents
        chunks = [doc.page_content for doc in docs]
        
        return chunks