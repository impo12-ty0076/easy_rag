"""
Text splitter utilities for Easy RAG System.
This module provides functionality for splitting text into chunks for vector databases.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
import importlib

class TextSplitter:
    """Base class for text splitters"""
    
    name = "Base Text Splitter"
    description = "Base text splitter class"
    
    @classmethod
    def get_name(cls) -> str:
        """Get the name of the text splitter"""
        return cls.name
    
    @classmethod
    def get_description(cls) -> str:
        """Get the description of the text splitter"""
        return cls.description
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for this text splitter"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int) -> Tuple[bool, str]:
        """
        Validate parameters for this text splitter
        Returns: (is_valid, error_message)
        """
        if chunk_size <= 0:
            return False, "Chunk size must be positive"
        
        if chunk_overlap < 0:
            return False, "Chunk overlap must be non-negative"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into chunks
        Returns: List of text chunks
        """
        raise NotImplementedError("Subclasses must implement this method")


class CharacterTextSplitter(TextSplitter):
    """Splitter for text based on character count with customizable separator"""
    
    name = "Character Text Splitter"
    description = "Splits text based on character count with customizable separator"
    
    # Common separators for easy selection
    COMMON_SEPARATORS = {
        "Paragraph": "\n\n",
        "Line": "\n",
        "Sentence": ". ",
        "Comma": ", ",
        "Space": " ",
        "None": ""
    }
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for character text splitter"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'separator': "\n\n",
            'is_separator_regex': False
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, separator: str = "\n\n", 
                           is_separator_regex: bool = False) -> Tuple[bool, str]:
        """Validate parameters for character text splitter"""
        if not (100 <= chunk_size <= 8000):
            return False, "Chunk size must be between 100 and 8000 characters"
        
        if not (0 <= chunk_overlap <= 500):
            return False, "Chunk overlap must be between 0 and 500 characters"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if separator is None:
            return False, "Separator cannot be None"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int, 
                  separator: str = "\n\n", is_separator_regex: bool = False) -> List[str]:
        """
        Split text into chunks based on character count and separator
        
        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: String separator to split on
            is_separator_regex: Whether the separator is a regex pattern
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # If separator is provided, first split by separator
        if separator:
            if is_separator_regex:
                import re
                splits = re.split(separator, text)
            else:
                splits = text.split(separator)
            
            # If the separator actually split the text
            if len(splits) > 1:
                chunks = []
                current_chunk = []
                current_length = 0
                
                for split in splits:
                    # Add the separator back to the split, except for the first one
                    if current_length > 0 and separator:
                        split_with_sep = separator + split if not is_separator_regex else split
                    else:
                        split_with_sep = split
                    
                    split_length = len(split_with_sep)
                    
                    # If adding this split would exceed the chunk size, process the current chunk
                    if current_length + split_length > chunk_size and current_chunk:
                        # Join the current chunk and add it to the list
                        chunks.append("".join(current_chunk))
                        
                        # Start a new chunk, with overlap from the previous chunk
                        if chunk_overlap > 0:
                            # Calculate how many characters to keep for overlap
                            overlap_text = "".join(current_chunk)
                            overlap_start = max(0, len(overlap_text) - chunk_overlap)
                            current_chunk = [overlap_text[overlap_start:]]
                            current_length = len(current_chunk[0])
                        else:
                            current_chunk = []
                            current_length = 0
                    
                    # Add the current split to the chunk
                    current_chunk.append(split_with_sep)
                    current_length += split_length
                
                # Add the last chunk if it exists
                if current_chunk:
                    chunks.append("".join(current_chunk))
                
                return chunks
        
        # Fall back to character-based splitting if separator doesn't work well
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Find the end of the chunk
            end = min(start + chunk_size, text_len)
            
            # If we're not at the end of the text and not at a whitespace, try to find a good break point
            if end < text_len and not text[end].isspace():
                # Look for the last whitespace within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Add the chunk to the list
            chunks.append(text[start:end])
            
            # Move the start position for the next chunk
            start = end - chunk_overlap
            
            # Make sure we're not stuck in an infinite loop
            if start >= end:
                start = end
        
        return chunks


class SemanticChunker(TextSplitter):
    """
    Semantic text splitter that splits text based on semantic similarity.
    Uses embeddings to determine natural breakpoints in text.
    
    의미론적 텍스트 분할기는 의미적 유사성에 기반하여 텍스트를 분할합니다.
    임베딩을 사용하여 텍스트의 자연스러운 분할점을 결정합니다.
    """
    
    name = "Semantic Chunker"
    description = "Splits text based on semantic similarity using embeddings"
    
    # Breakpoint threshold types
    THRESHOLD_TYPES = {
        "percentile": "Percentile",
        "standard_deviation": "Standard Deviation", 
        "interquartile": "Interquartile Range"
    }
    
    def __init__(self, embedding_model=None, breakpoint_threshold_type="percentile", 
                 breakpoint_threshold_amount=70):
        """
        Initialize SemanticChunker
        
        Args:
            embedding_model: Embedding model to use for semantic similarity
            breakpoint_threshold_type: Type of threshold ("percentile", "standard_deviation", "interquartile")
            breakpoint_threshold_amount: Threshold amount for breakpoint detection
        """
        self.embedding_model = embedding_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for semantic chunker"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_amount': 70,
            'sentence_split_regex': r'(?<=[.!?])\s+',
            'min_chunk_size': 100
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           breakpoint_threshold_type: str = "percentile",
                           breakpoint_threshold_amount: float = 70,
                           sentence_split_regex: str = r'(?<=[.!?])\s+',
                           min_chunk_size: int = 100) -> Tuple[bool, str]:
        """Validate parameters for semantic chunker"""
        if not (100 <= chunk_size <= 8000):
            return False, "Chunk size must be between 100 and 8000 characters"
        
        if not (0 <= chunk_overlap <= 500):
            return False, "Chunk overlap must be between 0 and 500 characters"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if breakpoint_threshold_type not in cls.THRESHOLD_TYPES:
            return False, f"Breakpoint threshold type must be one of: {list(cls.THRESHOLD_TYPES.keys())}"
        
        if breakpoint_threshold_type == "percentile":
            if not (0 <= breakpoint_threshold_amount <= 100):
                return False, "Percentile threshold must be between 0 and 100"
        elif breakpoint_threshold_type == "standard_deviation":
            if not (0.1 <= breakpoint_threshold_amount <= 5.0):
                return False, "Standard deviation threshold must be between 0.1 and 5.0"
        elif breakpoint_threshold_type == "interquartile":
            if not (0.1 <= breakpoint_threshold_amount <= 2.0):
                return False, "Interquartile threshold must be between 0.1 and 2.0"
        
        if not (50 <= min_chunk_size <= chunk_size):
            return False, f"Minimum chunk size must be between 50 and {chunk_size}"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int,
                  breakpoint_threshold_type: str = "percentile",
                  breakpoint_threshold_amount: float = 70,
                  sentence_split_regex: str = r'(?<=[.!?])\s+',
                  min_chunk_size: int = 100) -> List[str]:
        """
        Split text into chunks based on semantic similarity
        
        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            breakpoint_threshold_type: Type of threshold for breakpoint detection
            breakpoint_threshold_amount: Threshold amount for breakpoint detection
            sentence_split_regex: Regex pattern for splitting sentences
            min_chunk_size: Minimum size for a chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # If embedding model is not available, fall back to sentence-based splitting
        if not self.embedding_model:
            return self._fallback_sentence_split(text, chunk_size, chunk_overlap, 
                                               sentence_split_regex, min_chunk_size)
        
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text, sentence_split_regex)
            
            if len(sentences) <= 1:
                return [text] if len(text) >= min_chunk_size else []
            
            # Get embeddings for sentences
            embeddings = self._get_sentence_embeddings(sentences)
            
            if embeddings is None or len(embeddings) != len(sentences):
                # Fall back to sentence-based splitting if embeddings fail
                return self._fallback_sentence_split(text, chunk_size, chunk_overlap,
                                                   sentence_split_regex, min_chunk_size)
            
            # Calculate semantic distances between consecutive sentences
            distances = self._calculate_semantic_distances(embeddings)
            
            # Find breakpoints based on threshold type
            breakpoints = self._find_breakpoints(distances, breakpoint_threshold_type, 
                                               breakpoint_threshold_amount)
            
            # Create chunks based on breakpoints
            chunks = self._create_chunks_from_breakpoints(sentences, breakpoints, 
                                                        chunk_size, chunk_overlap, min_chunk_size)
            
            return chunks
            
        except Exception as e:
            # Fall back to sentence-based splitting on any error
            print(f"Error in semantic chunking: {str(e)}")
            return self._fallback_sentence_split(text, chunk_size, chunk_overlap,
                                               sentence_split_regex, min_chunk_size)
    
    def _split_into_sentences(self, text: str, sentence_split_regex: str) -> List[str]:
        """Split text into sentences using regex"""
        try:
            sentences = re.split(sentence_split_regex, text.strip())
            # Clean up sentences and remove empty ones
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except re.error:
            # If regex fails, split on common sentence endings
            sentences = re.split(r'[.!?]+\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for sentences using the embedding model"""
        try:
            if hasattr(self.embedding_model, 'embed_documents'):
                # For LangChain-style embedding models
                embeddings = self.embedding_model.embed_documents(sentences)
            elif hasattr(self.embedding_model, 'encode'):
                # For sentence-transformers style models
                embeddings = self.embedding_model.encode(sentences)
                embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            elif callable(self.embedding_model):
                # For custom embedding functions
                embeddings = [self.embedding_model(sentence) for sentence in sentences]
            else:
                return None
            
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return None
    
    def _calculate_semantic_distances(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine distances between consecutive sentence embeddings"""
        distances = []
        
        for i in range(len(embeddings) - 1):
            # Calculate cosine similarity between consecutive embeddings
            emb1 = np.array(embeddings[i])
            emb2 = np.array(embeddings[i + 1])
            
            # Normalize embeddings
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to distance (1 - similarity)
            distance = 1 - similarity
            distances.append(distance)
        
        return distances
    
    def _find_breakpoints(self, distances: List[float], threshold_type: str, 
                         threshold_amount: float) -> List[int]:
        """Find breakpoints based on semantic distances and threshold type"""
        if not distances:
            return []
        
        distances_array = np.array(distances)
        breakpoints = []
        
        if threshold_type == "percentile":
            threshold = np.percentile(distances_array, threshold_amount)
            breakpoints = [i for i, dist in enumerate(distances) if dist >= threshold]
            
        elif threshold_type == "standard_deviation":
            mean_dist = np.mean(distances_array)
            std_dist = np.std(distances_array)
            threshold = mean_dist + (threshold_amount * std_dist)
            breakpoints = [i for i, dist in enumerate(distances) if dist >= threshold]
            
        elif threshold_type == "interquartile":
            q1 = np.percentile(distances_array, 25)
            q3 = np.percentile(distances_array, 75)
            iqr = q3 - q1
            threshold = q3 + (threshold_amount * iqr)
            breakpoints = [i for i, dist in enumerate(distances) if dist >= threshold]
        
        return breakpoints
    
    def _create_chunks_from_breakpoints(self, sentences: List[str], breakpoints: List[int],
                                      chunk_size: int, chunk_overlap: int, 
                                      min_chunk_size: int) -> List[str]:
        """Create text chunks based on identified breakpoints"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Add sentence indices where we should break
        break_indices = set(breakpoints)
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (current_length + sentence_length > chunk_size and current_chunk) or i in break_indices:
                # Finalize current chunk if it meets minimum size
                if current_chunk and current_length >= min_chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if chunk_overlap > 0:
                        overlap_text = chunk_text[-chunk_overlap:] if len(chunk_text) > chunk_overlap else chunk_text
                        current_chunk = [overlap_text, sentence]
                        current_length = len(overlap_text) + sentence_length + 1  # +1 for space
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    # If current chunk is too small, just add the sentence
                    current_chunk.append(sentence)
                    current_length += sentence_length + (1 if current_chunk else 0)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)
        
        # Add the last chunk if it exists and meets minimum size
        if current_chunk and current_length >= min_chunk_size:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _fallback_sentence_split(self, text: str, chunk_size: int, chunk_overlap: int,
                               sentence_split_regex: str, min_chunk_size: int) -> List[str]:
        """Fallback method that splits text by sentences when semantic analysis fails"""
        sentences = self._split_into_sentences(text, sentence_split_regex)
        
        if not sentences:
            return [text] if len(text) >= min_chunk_size else []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Finalize current chunk if it meets minimum size
                if current_length >= min_chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if chunk_overlap > 0:
                        overlap_text = chunk_text[-chunk_overlap:] if len(chunk_text) > chunk_overlap else chunk_text
                        current_chunk = [overlap_text, sentence]
                        current_length = len(overlap_text) + sentence_length + 1
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    # If current chunk is too small, just add the sentence
                    current_chunk.append(sentence)
                    current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)
        
        # Add the last chunk if it exists and meets minimum size
        if current_chunk and current_length >= min_chunk_size:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Recursive Character Text Splitter that splits text using a list of characters in order.
    It attempts to split text until chunks are small enough, following a hierarchical approach:
    paragraphs -> sentences -> words -> characters.
    
    한국어 설명:
    RecursiveCharacterTextSplitter는 문자 목록을 매개변수로 받아 동작합니다.
    분할기는 청크가 충분히 작아질 때까지 주어진 문자 목록의 순서대로 텍스트를 분할하려고 시도합니다.
    단락 -> 문장 -> 단어 순서로 재귀적으로 분할합니다.
    """
    
    name = "Recursive Character Text Splitter"
    description = "Splits text recursively using a list of characters in order of preference"
    
    # Common separator sets for easy selection
    SEPARATOR_SETS = {
        "Default": ["\n\n", "\n", ". ", ", ", " ", ""],
        "Paragraphs": ["\n\n", "\n", ". "],
        "Sentences": [". ", "! ", "? ", "; ", ":", "\n", ", ", " "],
        "Words": [" ", "-", ""],
        "Characters": [""]
    }
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for recursive character text splitter"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'separators': cls.SEPARATOR_SETS["Default"],
            'is_separator_regex': False,
            'length_function': len
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           separators: List[str] = None, is_separator_regex: bool = False,
                           length_function: callable = len) -> Tuple[bool, str]:
        """Validate parameters for recursive character text splitter"""
        if not (100 <= chunk_size <= 8000):
            return False, "Chunk size must be between 100 and 8000 characters"
        
        if not (0 <= chunk_overlap <= 500):
            return False, "Chunk overlap must be between 0 and 500 characters"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if separators is not None and not isinstance(separators, list):
            return False, "Separators must be a list of strings"
        
        if not callable(length_function):
            return False, "Length function must be callable"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int, 
                  separators: List[str] = None, is_separator_regex: bool = False,
                  length_function: callable = len) -> List[str]:
        """
        Split text recursively using different separators
        
        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use in order of preference
            is_separator_regex: Whether the separators are regex patterns
            length_function: Function to calculate the length of text
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Use provided separators or default ones
        if separators is None:
            separators = self.SEPARATOR_SETS["Default"]
        
        # Use provided length function or default to len
        if length_function is None:
            length_function = len
        
        # Handle regex separators
        if is_separator_regex:
            return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators, length_function)
        else:
            return self._split_text_recursive(text, chunk_size, chunk_overlap, separators, length_function)
    
    def _split_text_recursive(self, text: str, chunk_size: int, chunk_overlap: int, 
                             separators: List[str], length_function: callable) -> List[str]:
        """Recursive implementation of text splitting using string separators"""
        # Base case: if we're at the last separator or the text is small enough
        if len(separators) == 0 or length_function(text) <= chunk_size:
            return [text]
        
        separator = separators[0]
        
        # If the separator is empty, we'll just do character splitting
        if separator == "":
            char_splitter = CharacterTextSplitter()
            return char_splitter.split_text(text, chunk_size, chunk_overlap)
        
        # Split the text by the current separator
        splits = text.split(separator)
        
        # If the split didn't work well (text wasn't split or too many tiny chunks)
        if len(splits) == 1 or max(length_function(s) for s in splits) < chunk_size / 2:
            # Try the next separator
            return self._split_text_recursive(text, chunk_size, chunk_overlap, separators[1:], length_function)
        
        # Process each split with the next level of separators
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # Add the separator back to the split, except for the first one
            if current_length > 0 and separator:
                split_with_sep = separator + split
            else:
                split_with_sep = split
            
            split_length = length_function(split_with_sep)
            
            # If adding this split would exceed the chunk size, process the current chunk
            if current_length + split_length > chunk_size and current_chunk:
                # Process the current chunk
                chunk_text = "".join(current_chunk)
                
                # If the chunk is still too big, recursively split it
                if length_function(chunk_text) > chunk_size:
                    chunks.extend(self._split_text_recursive(chunk_text, chunk_size, chunk_overlap, separators[1:], length_function))
                else:
                    chunks.append(chunk_text)
                
                # Start a new chunk, with overlap from the previous chunk
                overlap_start = max(0, len("".join(current_chunk)) - chunk_overlap)
                current_chunk = [chunk_text[overlap_start:]]
                current_length = length_function(current_chunk[0])
            
            # Add the current split to the chunk
            current_chunk.append(split_with_sep)
            current_length += split_length
        
        # Process the last chunk if it exists
        if current_chunk:
            chunk_text = "".join(current_chunk)
            
            # If the chunk is still too big, recursively split it
            if length_function(chunk_text) > chunk_size:
                chunks.extend(self._split_text_recursive(chunk_text, chunk_size, chunk_overlap, separators[1:], length_function))
            else:
                chunks.append(chunk_text)
        
        return chunks
        
    def _split_text_recursive_regex(self, text: str, chunk_size: int, chunk_overlap: int, 
                                   separators: List[str], length_function: callable) -> List[str]:
        """Recursive implementation of text splitting using regex separators"""
        import re
        
        # Base case: if we're at the last separator or the text is small enough
        if len(separators) == 0 or length_function(text) <= chunk_size:
            return [text]
        
        separator_pattern = separators[0]
        
        # If the separator is empty, we'll just do character splitting
        if separator_pattern == "":
            char_splitter = CharacterTextSplitter()
            return char_splitter.split_text(text, chunk_size, chunk_overlap)
        
        try:
            # Split the text by the current separator pattern
            splits = re.split(separator_pattern, text)
            
            # If the split didn't work well (text wasn't split or too many tiny chunks)
            if len(splits) == 1 or max(length_function(s) for s in splits) < chunk_size / 2:
                # Try the next separator
                return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators[1:], length_function)
            
            # Process each split with the next level of separators
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Find all matches to reconstruct the text with separators
            matches = list(re.finditer(separator_pattern, text))
            
            for i, split in enumerate(splits):
                # For all but the first split, add the separator
                if i > 0 and i-1 < len(matches):
                    separator_match = matches[i-1]
                    separator_text = separator_match.group(0)
                    split_with_sep = separator_text + split
                else:
                    split_with_sep = split
                
                split_length = length_function(split_with_sep)
                
                # If adding this split would exceed the chunk size, process the current chunk
                if current_length + split_length > chunk_size and current_chunk:
                    # Process the current chunk
                    chunk_text = "".join(current_chunk)
                    
                    # If the chunk is still too big, recursively split it
                    if length_function(chunk_text) > chunk_size:
                        chunks.extend(self._split_text_recursive_regex(chunk_text, chunk_size, chunk_overlap, separators[1:], length_function))
                    else:
                        chunks.append(chunk_text)
                    
                    # Start a new chunk, with overlap from the previous chunk
                    overlap_start = max(0, len("".join(current_chunk)) - chunk_overlap)
                    current_chunk = [chunk_text[overlap_start:]]
                    current_length = length_function(current_chunk[0])
                
                # Add the current split to the chunk
                current_chunk.append(split_with_sep)
                current_length += split_length
            
            # Process the last chunk if it exists
            if current_chunk:
                chunk_text = "".join(current_chunk)
                
                # If the chunk is still too big, recursively split it
                if length_function(chunk_text) > chunk_size:
                    chunks.extend(self._split_text_recursive_regex(chunk_text, chunk_size, chunk_overlap, separators[1:], length_function))
                else:
                    chunks.append(chunk_text)
            
            return chunks
            
        except re.error:
            # If there's an error with the regex, try the next separator
            return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators[1:], length_function)


class SemanticChunker(TextSplitter):
    """
    Semantic Chunker that splits text based on semantic similarity.
    
    This splitter uses embeddings to determine semantic breakpoints in text.
    It can use different methods to determine where to split:
    - percentile: Splits based on a percentile of all differences between adjacent sentences
    - standard_deviation: Splits based on standard deviation from the mean difference
    - interquartile: Splits based on interquartile range
    
    한국어 설명:
    SemanticChunker는 텍스트를 의미론적 유사성에 기반하여 분할합니다.
    임베딩을 사용하여 텍스트의 의미적 단절점을 결정합니다.
    다음과 같은 다양한 방법으로 분할 지점을 결정할 수 있습니다:
    - percentile: 인접한 문장 간의 모든 차이의 백분위수를 기준으로 분할
    - standard_deviation: 평균 차이에서 표준 편차를 기준으로 분할
    - interquartile: 사분위수 범위를 기준으로 분할
    """
    
    name = "Semantic Chunker"
    description = "Splits text based on semantic similarity using embeddings"
    
    # Available threshold types
    THRESHOLD_TYPES = ["percentile", "standard_deviation", "interquartile"]
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for semantic chunker"""
        return {
            'chunk_size': 1000,  # Not directly used but kept for compatibility
            'chunk_overlap': 0,  # Not directly used but kept for compatibility
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_amount': 70,
            'sentence_separator': '.',
            'embedding_model': None  # This should be provided by the user
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           breakpoint_threshold_type: str = 'percentile',
                           breakpoint_threshold_amount: float = 70,
                           sentence_separator: str = '.',
                           embedding_model: Any = None) -> Tuple[bool, str]:
        """Validate parameters for semantic chunker"""
        # chunk_size and chunk_overlap are not directly used but kept for compatibility
        
        if breakpoint_threshold_type not in cls.THRESHOLD_TYPES:
            return False, f"Threshold type must be one of {cls.THRESHOLD_TYPES}"
        
        if breakpoint_threshold_type == 'percentile' and not (0 <= breakpoint_threshold_amount <= 100):
            return False, "Percentile threshold must be between 0 and 100"
        
        if breakpoint_threshold_type == 'standard_deviation' and breakpoint_threshold_amount <= 0:
            return False, "Standard deviation threshold must be positive"
        
        if breakpoint_threshold_type == 'interquartile' and breakpoint_threshold_amount <= 0:
            return False, "Interquartile threshold must be positive"
        
        if not sentence_separator:
            return False, "Sentence separator cannot be empty"
        
        if embedding_model is None:
            return False, "Embedding model must be provided"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None,
                  breakpoint_threshold_type: str = 'percentile',
                  breakpoint_threshold_amount: float = 70,
                  sentence_separator: str = '.',
                  embedding_model: Any = None) -> List[str]:
        """
        Split text based on semantic similarity
        
        Args:
            text: The text to split
            chunk_size: Not directly used but kept for compatibility
            chunk_overlap: Not directly used but kept for compatibility
            breakpoint_threshold_type: Method to determine breakpoints ('percentile', 'standard_deviation', 'interquartile')
            breakpoint_threshold_amount: Threshold value for the selected method
            sentence_separator: Character or string used to split text into sentences
            embedding_model: Model to use for creating embeddings
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if embedding_model is None:
            raise ValueError("Embedding model must be provided")
        
        # Split text into sentences
        sentences = [s.strip() for s in text.split(sentence_separator) if s.strip()]
        if not sentences:
            return [text]
        
        # If there's only one sentence, return it as is
        if len(sentences) == 1:
            return sentences
        
        try:
            # Get embeddings for each sentence
            embeddings = self._get_embeddings(sentences, embedding_model)
            
            # Calculate cosine similarities between adjacent sentences
            similarities = self._calculate_similarities(embeddings)
            
            # Calculate differences (1 - similarity) to find semantic breakpoints
            differences = [1 - sim for sim in similarities]
            
            # Determine breakpoints based on the selected method
            breakpoints = self._find_breakpoints(
                differences, 
                breakpoint_threshold_type, 
                breakpoint_threshold_amount
            )
            
            # Create chunks based on breakpoints
            chunks = self._create_chunks(sentences, breakpoints, sentence_separator)
            
            return chunks
            
        except Exception as e:
            # If there's an error in semantic chunking, fall back to simple sentence splitting
            print(f"Error in semantic chunking: {str(e)}. Falling back to sentence splitting.")
            return [sentence_separator.join(sentences)]
    
    def _get_embeddings(self, sentences: List[str], embedding_model: Any) -> List[List[float]]:
        """Get embeddings for a list of sentences"""
        # This is a simplified implementation
        # In a real implementation, you would use the provided embedding model
        try:
            # Try to use the embedding model's embed_documents method
            if hasattr(embedding_model, 'embed_documents'):
                return embedding_model.embed_documents(sentences)
            # Fall back to embed_query if embed_documents is not available
            elif hasattr(embedding_model, 'embed_query'):
                return [embedding_model.embed_query(sentence) for sentence in sentences]
            else:
                raise AttributeError("Embedding model must have embed_documents or embed_query method")
        except Exception as e:
            raise ValueError(f"Failed to generate embeddings: {str(e)}")
    
    def _calculate_similarities(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between adjacent embeddings"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i in range(len(embeddings) - 1):
            # Convert to numpy arrays and reshape for cosine_similarity
            emb1 = np.array(embeddings[i]).reshape(1, -1)
            emb2 = np.array(embeddings[i + 1]).reshape(1, -1)
            
            # Calculate cosine similarity
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(sim)
        
        return similarities
    
    def _find_breakpoints(self, differences: List[float], 
                         threshold_type: str, 
                         threshold_amount: float) -> List[int]:
        """Find breakpoints based on the selected method"""
        import numpy as np
        
        breakpoints = []
        
        if threshold_type == 'percentile':
            # Find breakpoints based on percentile
            threshold = np.percentile(differences, threshold_amount)
            for i, diff in enumerate(differences):
                if diff >= threshold:
                    breakpoints.append(i + 1)  # +1 because we want to break after this sentence
        
        elif threshold_type == 'standard_deviation':
            # Find breakpoints based on standard deviation
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            threshold = mean_diff + (threshold_amount * std_diff)
            
            for i, diff in enumerate(differences):
                if diff >= threshold:
                    breakpoints.append(i + 1)
        
        elif threshold_type == 'interquartile':
            # Find breakpoints based on interquartile range
            q1 = np.percentile(differences, 25)
            q3 = np.percentile(differences, 75)
            iqr = q3 - q1
            threshold = q3 + (threshold_amount * iqr)
            
            for i, diff in enumerate(differences):
                if diff >= threshold:
                    breakpoints.append(i + 1)
        
        return breakpoints
    
    def _create_chunks(self, sentences: List[str], breakpoints: List[int], 
                      sentence_separator: str) -> List[str]:
        """Create chunks based on breakpoints"""
        chunks = []
        start_idx = 0
        
        # Add sentence separator back to sentences
        sentences_with_sep = [s + sentence_separator for s in sentences[:-1]] + [sentences[-1]]
        
        # Create chunks based on breakpoints
        for bp in breakpoints:
            chunk = ''.join(sentences_with_sep[start_idx:bp])
            chunks.append(chunk)
            start_idx = bp
        
        # Add the last chunk
        if start_idx < len(sentences):
            chunk = ''.join(sentences_with_sep[start_idx:])
            chunks.append(chunk)
        
        return chunks


class RecursiveTextSplitter(TextSplitter):
    """Recursive splitter that tries different separators in order"""
    
    name = "Recursive Text Splitter"
    description = "Intelligently splits text based on multiple separators in order of preference"
    
    # Common separator sets for easy selection
    SEPARATOR_SETS = {
        "Default": ["\n\n", "\n", ". ", ", ", " ", ""],
        "Paragraphs": ["\n\n", "\n", ". "],
        "Sentences": [". ", "! ", "? ", "; ", ":", "\n", ", ", " "],
        "Words": [" ", "-", ""],
        "Characters": [""]
    }
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for recursive text splitter"""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'separators': cls.SEPARATOR_SETS["Default"],
            'is_separator_regex': False
        }
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           separators: List[str] = None, is_separator_regex: bool = False) -> Tuple[bool, str]:
        """Validate parameters for recursive text splitter"""
        if not (100 <= chunk_size <= 8000):
            return False, "Chunk size must be between 100 and 8000 characters"
        
        if not (0 <= chunk_overlap <= 500):
            return False, "Chunk overlap must be between 0 and 500 characters"
        
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if separators is not None and not isinstance(separators, list):
            return False, "Separators must be a list of strings"
        
        return True, ""
    
    def split_text(self, text: str, chunk_size: int, chunk_overlap: int, 
                  separators: List[str] = None, is_separator_regex: bool = False) -> List[str]:
        """
        Split text recursively using different separators
        
        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use in order of preference
            is_separator_regex: Whether the separators are regex patterns
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Use provided separators or default ones
        if separators is None:
            separators = self.SEPARATOR_SETS["Default"]
        
        # Handle regex separators
        if is_separator_regex:
            return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators)
        else:
            return self._split_text_recursive(text, chunk_size, chunk_overlap, separators)
    
    def _split_text_recursive(self, text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        """Recursive implementation of text splitting using string separators"""
        # Base case: if we're at the last separator or the text is small enough
        if len(separators) == 0 or len(text) <= chunk_size:
            return [text]
        
        separator = separators[0]
        
        # If the separator is empty, we'll just do character splitting
        if separator == "":
            char_splitter = CharacterTextSplitter()
            return char_splitter.split_text(text, chunk_size, chunk_overlap)
        
        # Split the text by the current separator
        splits = text.split(separator)
        
        # If the split didn't work well (text wasn't split or too many tiny chunks)
        if len(splits) == 1 or max(len(s) for s in splits) < chunk_size / 2:
            # Try the next separator
            return self._split_text_recursive(text, chunk_size, chunk_overlap, separators[1:])
        
        # Process each split with the next level of separators
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # Add the separator back to the split, except for the first one
            if current_length > 0 and separator:
                split_with_sep = separator + split
            else:
                split_with_sep = split
            
            split_length = len(split_with_sep)
            
            # If adding this split would exceed the chunk size, process the current chunk
            if current_length + split_length > chunk_size and current_chunk:
                # Process the current chunk
                chunk_text = "".join(current_chunk)
                
                # If the chunk is still too big, recursively split it
                if len(chunk_text) > chunk_size:
                    chunks.extend(self._split_text_recursive(chunk_text, chunk_size, chunk_overlap, separators[1:]))
                else:
                    chunks.append(chunk_text)
                
                # Start a new chunk, with overlap from the previous chunk
                overlap_start = max(0, len("".join(current_chunk)) - chunk_overlap)
                current_chunk = [chunk_text[overlap_start:]]
                current_length = len(current_chunk[0])
            
            # Add the current split to the chunk
            current_chunk.append(split_with_sep)
            current_length += split_length
        
        # Process the last chunk if it exists
        if current_chunk:
            chunk_text = "".join(current_chunk)
            
            # If the chunk is still too big, recursively split it
            if len(chunk_text) > chunk_size:
                chunks.extend(self._split_text_recursive(chunk_text, chunk_size, chunk_overlap, separators[1:]))
            else:
                chunks.append(chunk_text)
        
        return chunks
        
    def _split_text_recursive_regex(self, text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        """Recursive implementation of text splitting using regex separators"""
        import re
        
        # Base case: if we're at the last separator or the text is small enough
        if len(separators) == 0 or len(text) <= chunk_size:
            return [text]
        
        separator_pattern = separators[0]
        
        # If the separator is empty, we'll just do character splitting
        if separator_pattern == "":
            char_splitter = CharacterTextSplitter()
            return char_splitter.split_text(text, chunk_size, chunk_overlap)
        
        try:
            # Split the text by the current separator pattern
            splits = re.split(separator_pattern, text)
            
            # If the split didn't work well (text wasn't split or too many tiny chunks)
            if len(splits) == 1 or max(len(s) for s in splits) < chunk_size / 2:
                # Try the next separator
                return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators[1:])
            
            # Process each split with the next level of separators
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Find all matches to reconstruct the text with separators
            matches = list(re.finditer(separator_pattern, text))
            
            for i, split in enumerate(splits):
                # For all but the first split, add the separator
                if i > 0 and i-1 < len(matches):
                    separator_match = matches[i-1]
                    separator_text = separator_match.group(0)
                    split_with_sep = separator_text + split
                else:
                    split_with_sep = split
                
                split_length = len(split_with_sep)
                
                # If adding this split would exceed the chunk size, process the current chunk
                if current_length + split_length > chunk_size and current_chunk:
                    # Process the current chunk
                    chunk_text = "".join(current_chunk)
                    
                    # If the chunk is still too big, recursively split it
                    if len(chunk_text) > chunk_size:
                        chunks.extend(self._split_text_recursive_regex(chunk_text, chunk_size, chunk_overlap, separators[1:]))
                    else:
                        chunks.append(chunk_text)
                    
                    # Start a new chunk, with overlap from the previous chunk
                    overlap_start = max(0, len("".join(current_chunk)) - chunk_overlap)
                    current_chunk = [chunk_text[overlap_start:]]
                    current_length = len(current_chunk[0])
                
                # Add the current split to the chunk
                current_chunk.append(split_with_sep)
                current_length += split_length
            
            # Process the last chunk if it exists
            if current_chunk:
                chunk_text = "".join(current_chunk)
                
                # If the chunk is still too big, recursively split it
                if len(chunk_text) > chunk_size:
                    chunks.extend(self._split_text_recursive_regex(chunk_text, chunk_size, chunk_overlap, separators[1:]))
                else:
                    chunks.append(chunk_text)
            
            return chunks
            
        except re.error:
            # If there's an error with the regex, try the next separator
            return self._split_text_recursive_regex(text, chunk_size, chunk_overlap, separators[1:])


# Dictionary of available text splitter classes
AVAILABLE_TEXT_SPLITTERS = {
    'character': CharacterTextSplitter,
    'recursive_character': RecursiveCharacterTextSplitter,
    'semantic': SemanticChunker,
    'code': CodeTextSplitterWrapper,
    'recursive_character': RecursiveCharacterTextSplitter,
    'semantic': SemanticChunker,
    # 추가 text splitter들을 위한 공간
    'code': None,       # Will be dynamically loaded
    'markdown_header': None,  # MarkdownHeaderTextSplitter
    'html_header': None,      # HTMLHeaderTextSplitter
    'recursive_json': None    # RecursiveJsonSplitter
}


def get_available_text_splitters_metadata() -> List[Dict[str, Any]]:
    """Get a list of all available text splitters with their metadata"""
    splitters = []
    
    for splitter_id, splitter_class in AVAILABLE_TEXT_SPLITTERS.items():
        if splitter_class is not None:  # Only include implemented splitters
            splitters.append({
                'id': splitter_id,
                'name': splitter_class.get_name(),
                'description': splitter_class.get_description(),
                'default_parameters': splitter_class.get_default_parameters()
            })
    
    return splitters
    return splitters


def get_text_splitter(splitter_type: str) -> TextSplitter:
    """Get a text splitter instance by type"""
    if splitter_type not in AVAILABLE_TEXT_SPLITTERS:
        raise ValueError(f"Unknown text splitter type: {splitter_type}")
    
    return AVAILABLE_TEXT_SPLITTERS[splitter_type]()


def split_text(text: str, splitter_type: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text using the specified splitter type and parameters"""
    splitter = get_text_splitter(splitter_type)
    
    # Validate parameters
    is_valid, error_message = splitter.validate_parameters(chunk_size, chunk_overlap)
    if not is_valid:
        raise ValueError(error_message)
    
    return splitter.split_text(text, chunk_size, chunk_overlap)


# Try to dynamically load the CodeTextSplitter
try:
    from easy_rag.utils.code_text_splitter import CodeTextSplitter
    AVAILABLE_TEXT_SPLITTERS['code'] = CodeTextSplitter
except ImportError:
    # If langchain_text_splitters is not installed, CodeTextSplitter will remain None
    pass

# Import the CodeTextSplitter
from easy_rag.utils.code_text_splitter import CodeTextSplitter

# This section is now handled by the get_available_text_splitters function at the end of the file

def create_text_splitter(splitter_type):
    """
    Create a text splitter instance based on type
    
    Args:
        splitter_type: Type of text splitter to create
        
    Returns:
        TextSplitter: Instance of the requested text splitter
    """
    splitters = get_available_text_splitters()
    
    if splitter_type not in splitters:
        raise ValueError(f"Unknown text splitter type: {splitter_type}")
    
    return splitters[splitter_type]()
c
lass CodeTextSplitterWrapper(TextSplitter):
    """
    Code Text Splitter that splits code using language-specific separators.
    This is a wrapper around the CodeTextSplitter class from easy_rag.utils.code_text_splitter.
    
    코드 텍스트 분할기는 언어별 구분자를 사용하여 코드를 분할합니다.
    이는 easy_rag.utils.code_text_splitter의 CodeTextSplitter 클래스의 래퍼입니다.
    """
    
    name = "Code Text Splitter"
    description = "Splits code based on language-specific syntax"
    
    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """Get default parameters for code text splitter"""
        from easy_rag.utils.code_text_splitter import CodeTextSplitter
        return CodeTextSplitter.get_default_parameters()
    
    @classmethod
    def validate_parameters(cls, chunk_size: int, chunk_overlap: int, 
                           language: str = 'PYTHON') -> Tuple[bool, str]:
        """Validate parameters for code text splitter"""
        from easy_rag.utils.code_text_splitter import CodeTextSplitter
        return CodeTextSplitter.validate_parameters(chunk_size, chunk_overlap, language)
    
    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported programming languages"""
        from easy_rag.utils.code_text_splitter import CodeTextSplitter
        return CodeTextSplitter.get_supported_languages()
    
    @classmethod
    def get_separators_for_language(cls, language: str) -> List[str]:
        """Get the separators used for a specific language"""
        from easy_rag.utils.code_text_splitter import CodeTextSplitter
        return CodeTextSplitter.get_separators_for_language(language)
    
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
        from easy_rag.utils.code_text_splitter import CodeTextSplitter
        
        code_splitter = CodeTextSplitter()
        return code_splitter.split_text(text, chunk_size, chunk_overlap, language)


def get_available_text_splitters() -> Dict[str, TextSplitter]:
    """
    Get all available text splitters (instances)
    
    Returns:
        Dictionary of text splitter name to instance
    """
    return {
        splitter_id: splitter_class() 
        for splitter_id, splitter_class in AVAILABLE_TEXT_SPLITTERS.items()
        if splitter_class is not None
    }


def get_language_enum_values() -> List[str]:
    """
    Get all available language enum values for code text splitting
    
    Returns:
        List of language enum values
    """
    try:
        from langchain_text_splitters import Language
        return [e.value for e in Language]
    except ImportError:
        return []


def get_language_separators(language_value: str) -> List[str]:
    """
    Get the separators used for a specific language
    
    Args:
        language_value: The language value from Language enum
        
    Returns:
        List of separators used for the language
    """
    try:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        
        # Convert string value to Language enum
        for lang in Language:
            if lang.value == language_value:
                return RecursiveCharacterTextSplitter.get_separators_for_language(lang)
        
        return []
    except ImportError:
        return []


# Example of using RecursiveCharacterTextSplitter with a specific language
def split_code_example(code_text: str, language_value: str, chunk_size: int = 128, chunk_overlap: int = 0) -> List[str]:
    """
    Example function to split code using RecursiveCharacterTextSplitter with a specific language
    
    Args:
        code_text: The code text to split
        language_value: The language value from Language enum
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of code chunks
    """
    try:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        
        # Convert string value to Language enum
        language_enum = None
        for lang in Language:
            if lang.value == language_value:
                language_enum = lang
                break
        
        if language_enum is None:
            # Default to Python if language not found
            language_enum = Language.PYTHON
        
        # Create a RecursiveCharacterTextSplitter with language-specific settings
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_enum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split the code
        docs = splitter.create_documents([code_text])
        
        # Extract the content from the documents
        chunks = [doc.page_content for doc in docs]
        
        return chunks
    except ImportError:
        # Fall back to character splitting if langchain_text_splitters is not available
        char_splitter = CharacterTextSplitter()
        return char_splitter.split_text(code_text, chunk_size, chunk_overlap)