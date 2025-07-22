# Requirements Document

## Introduction

The Easy RAG System is a user-friendly application designed to enable non-developers to create and use Retrieval-Augmented Generation (RAG) systems. The application provides a simple interface for managing documents, creating vector databases, configuring retrieval strategies, and interacting with various language models. Users can process local documents, convert them into vector representations, and query them using different language models without requiring programming knowledge.

## Requirements

### 1. Document Management

**User Story:** As a non-technical user, I want to manage my document collection, so that I can organize the content that will be used in my RAG system.

#### Acceptance Criteria

1. WHEN the user starts the application THEN the system SHALL provide a document management interface.
2. WHEN the user first runs the application THEN the system SHALL prompt for initial settings including local document storage path.
3. WHEN the user selects a document or folder THEN the system SHALL display its metadata (file type, size, last modified date).
4. WHEN the user adds a document THEN the system SHALL validate if the file format is supported.
5. WHEN the user removes a document THEN the system SHALL confirm before deletion from the document collection.
6. WHEN the user navigates to the document management screen THEN the system SHALL display all currently added documents.

### 2. Vector Database Creation

**User Story:** As a non-technical user, I want to convert my documents into vector databases, so that I can perform semantic searches on them.

#### Acceptance Criteria

1. WHEN the user navigates to the vector DB creation screen THEN the system SHALL display available document loaders.
2. WHEN the user selects a document loader THEN the system SHALL allow selection of target documents or folders.
3. WHEN the user selects a document loader THEN the system SHALL automatically install required dependencies.
4. WHEN the user attempts to create a vector database THEN the system SHALL validate that all required inputs are provided.
5. WHEN the system detects missing dependencies THEN the system SHALL automatically install them via terminal commands.
6. WHEN the user creates a vector database THEN the system SHALL save metadata about the creation process for future reference.
7. WHEN the vector database creation is complete THEN the system SHALL notify the user of successful completion.
8. WHEN the user selects a folder or files THEN the system SHALL detect file extensions and automatically suggest appropriate document loaders.
9. WHEN the system detects multiple file types THEN the system SHALL allow selection of multiple document loaders for different file types.
10. WHEN the user selects a folder THEN the system SHALL process all supported files within that folder using appropriate loaders.

### 3. Text Splitting Configuration

**User Story:** As a non-technical user, I want to configure text splitting strategies, so that my documents are chunked optimally for retrieval.

#### Acceptance Criteria

1. WHEN the user navigates to the text splitting configuration THEN the system SHALL display available chunking strategies.
2. WHEN the user configures text splitting THEN the system SHALL allow specification of chunk size.
3. WHEN the user configures text splitting THEN the system SHALL allow specification of chunk overlap.
4. WHEN the user selects a chunking strategy THEN the system SHALL provide appropriate default values.
5. WHEN the user changes chunking parameters THEN the system SHALL validate the inputs for reasonable ranges.
6. WHEN the user completes text splitting configuration THEN the system SHALL save these settings for the current vector database creation process.

### 4. Embedding Model Selection

**User Story:** As a non-technical user, I want to select embedding models and vector stores, so that I can create vector representations of my documents.

#### Acceptance Criteria

1. WHEN the user navigates to embedding model selection THEN the system SHALL display available embedding models.
2. WHEN the user navigates to vector store selection THEN the system SHALL display available vector stores (Chroma, FAISS, Pinecone).
3. WHEN the user selects an API-based embedding model THEN the system SHALL check for required API keys in the .env file.
4. WHEN required API keys are missing THEN the system SHALL display instructions for adding them.
5. WHEN the user selects a Hugging Face embedding model THEN the system SHALL download the model automatically, Or can use a model that was previously downloaded locally.
6. WHEN the user completes embedding model and vector store selection THEN the system SHALL proceed with vector database creation.
7. WHEN the vector database creation process is running THEN the system SHALL display progress information.

### 5. Retriever Configuration

**User Story:** As a non-technical user, I want to configure retrieval strategies, so that I can optimize how documents are retrieved for my queries.

#### Acceptance Criteria

1. WHEN the user navigates to the RAG usage screen THEN the system SHALL load the embedding model information used for the selected vector database.
2. WHEN the user navigates to retriever configuration THEN the system SHALL display available retriever types.
3. WHEN the user selects a retriever type THEN the system SHALL display appropriate configuration options for that retriever.
4. WHEN the user configures reranking THEN the system SHALL allow selection of a reranking LLM.
5. WHEN the user configures retrieval THEN the system SHALL allow specification of the number of source chunks to retrieve.
6. WHEN the user selects Hybrid Search THEN the system must have hybrid search-related code running.
7. WHEN the user completes retriever configuration THEN the system SHALL save these settings for the current RAG session.

### 6. LLM Selection and Query Interface

**User Story:** As a non-technical user, I want to select language models and query my vector database, so that I can get RAG-enhanced responses.

#### Acceptance Criteria

1. WHEN the user navigates to the LLM selection screen THEN the system SHALL display available language models.
2. WHEN the user selects an API-based LLM THEN the system SHALL check for required API keys in the .env file.
3. WHEN required API keys are missing THEN the system SHALL display the LLM as unavailable.
4. WHEN the user chooses the Hugging Face quantization model THEN the system automatically downloads the model, or can use a model that was previously downloaded locally.
5. WHEN the user enters a query THEN the system SHALL process it using the configured retriever and LLM.
6. WHEN the system generates a response THEN the system SHALL display both the response and the retrieved context chunks.
7. WHEN the user changes configuration settings THEN the system SHALL apply them to subsequent queries.
8. WHEN the system encounters an error during query processing THEN the system SHALL display a user-friendly error message.

### 7. System Installation and Dependencies

**User Story:** As a non-technical user, I want the system to handle dependencies automatically, so that I don't need to manually install libraries.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL check for required core dependencies.
2. WHEN core dependencies are missing THEN the system SHALL install them automatically.
3. WHEN a specific feature requires additional dependencies THEN the system SHALL install them when that feature is accessed.
4. WHEN dependency installation fails THEN the system SHALL provide clear error messages and recovery instructions.
5. WHEN the user requests it THEN the system SHALL generate a requirements.txt file with all dependencies.
6. WHEN the system installs dependencies THEN the system SHALL show progress in a user-friendly way.