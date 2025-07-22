# Easy RAG System - Usage Tutorials

## Introduction

This document provides step-by-step tutorials for using the key features of the Easy RAG System. Each tutorial is designed to guide you through a specific workflow, from document management to querying your RAG system.

## Table of Contents

1. [Managing Documents](#managing-documents)
2. [Creating Vector Databases](#creating-vector-databases)
3. [Configuring Retrievers](#configuring-retrievers)
4. [Querying Your Documents](#querying-your-documents)
5. [Managing API Keys and Settings](#managing-api-keys-and-settings)

## Managing Documents

### Uploading Documents

1. Navigate to the **Documents** page from the main navigation menu
2. Click the **Upload Documents** button
3. Either:
   - Select files using the file browser
   - Drag and drop files onto the upload area
4. Wait for the upload to complete
5. Your documents will appear in the document list

**Supported File Formats:**
- PDF (.pdf)
- Text (.txt)
- Microsoft Word (.docx, .doc)
- HTML (.html, .htm)
- Markdown (.md)
- CSV (.csv)
- And more

### Viewing Document Details

1. In the document list, click on a document's name
2. The document details page will show:
   - Basic metadata (file size, type, upload date)
   - A preview of the document content
   - Available actions

### Deleting Documents

1. In the document list, find the document you want to delete
2. Click the **Delete** button (trash icon)
3. Confirm the deletion when prompted

**Note:** Deleting a document will not automatically remove it from any vector databases it's already part of.

## Creating Vector Databases

### Starting the Creation Process

1. Navigate to the **Vector Databases** page
2. Click the **Create New Vector Database** button
3. Follow the step-by-step wizard

### Step 1: Select Documents

1. Choose the documents you want to include in your vector database
2. You can select individual files or entire folders
3. Click **Next** to continue

### Step 2: Choose Document Loader

1. Select the appropriate document loader for your file types
2. The system will recommend loaders based on your selected documents
3. Click **Next** to continue

### Step 3: Configure Text Splitting

1. Choose a text splitting strategy:
   - **Character Text Splitter**: Splits by character count
   - **Token Text Splitter**: Splits by token count (better for LLMs)
   - **Recursive Character Text Splitter**: Intelligently splits by structure
2. Set the chunk size (how large each chunk should be)
3. Set the chunk overlap (how much chunks should overlap)
4. Click **Next** to continue

**Recommended Settings:**
- For general use: Recursive Character Text Splitter with 1000 chunk size and 200 overlap
- For code or technical documents: Token Text Splitter with 500 chunk size and 50 overlap

### Step 4: Select Embedding Model

1. Choose an embedding model:
   - **Local Models**: Run on your computer (e.g., BAAI/bge-small-en)
   - **API-based Models**: Higher quality but require API keys (e.g., OpenAI)
2. If using an API-based model, ensure you've added the required API key
3. Click **Next** to continue

### Step 5: Choose Vector Store

1. Select a vector store:
   - **Chroma**: Good all-around performance
   - **FAISS**: Optimized for large collections
   - **Pinecone**: Cloud-based, requires API key
2. Configure any vector store-specific settings
3. Click **Create** to start the vector database creation process

### Monitoring Creation Progress

1. The system will show a progress bar during creation
2. You can navigate away and come back later; the process will continue
3. Once complete, you'll see a success message and the new vector database in your list

## Configuring Retrievers

### Accessing Retriever Configuration

1. Navigate to the **Retrievers** page
2. Select a vector database to configure retrievers for
3. Click **Configure Retriever**

### Choosing Retriever Type

1. Select a retriever type:
   - **Similarity Search**: Basic semantic search
   - **MMR (Maximum Marginal Relevance)**: Balances relevance with diversity
   - **Hybrid Search**: Combines semantic and keyword search
2. Configure retriever-specific settings

### Setting Retrieval Parameters

1. Set the number of chunks to retrieve for each query
2. Configure any advanced options:
   - For MMR: Set the diversity parameter
   - For Hybrid Search: Set the alpha parameter (balance between semantic and keyword search)
3. Optionally enable reranking for improved relevance
4. Save your configuration

## Querying Your Documents

### Basic Querying

1. Navigate to the **Query** page
2. Select the vector database you want to query
3. Choose the configured retriever to use
4. Select a language model (LLM):
   - **Local Models**: Run on your computer
   - **API-based Models**: Higher quality but require API keys
5. Enter your question in the query box
6. Click **Submit** to get an answer

### Understanding Results

The query results include:

1. **Answer**: The generated response to your question
2. **Source Chunks**: The document chunks that were used to generate the answer
3. **Relevance Scores**: How relevant each chunk is to your question

### Tips for Better Questions

- Be specific in your questions
- Use clear, concise language
- Break complex questions into simpler ones
- Experiment with different retriever settings for different types of questions

## Managing API Keys and Settings

### Adding API Keys

1. Navigate to the **Settings** page
2. In the API Keys section, enter your keys for:
   - OpenAI
   - Hugging Face
   - Pinecone
   - Other services as needed
3. Click **Save** to store your API keys securely in the .env file

### Configuring Storage Paths

1. Navigate to the **Settings** page
2. In the Storage Paths section, you can modify:
   - Document Storage Path
   - Vector Database Path
3. Click **Save** to apply changes

### Managing Dependencies

1. Navigate to the **Dependencies** page
2. View installed dependencies and their versions
3. Install missing dependencies if needed
4. Generate a requirements.txt file for your current configuration

## Advanced Usage

### Using Local LLMs

For privacy or offline use, you can use local LLMs:

1. Navigate to the **Query** page
2. In the LLM selection, choose a local model (e.g., Llama-2-7b-chat)
3. The first time you select a model, it will be downloaded automatically
4. Subsequent queries will use the downloaded model

### Creating Custom Workflows

You can create custom workflows by combining different components:

1. Create multiple vector databases with different configurations
2. Configure different retrievers for different types of queries
3. Use different LLMs based on your needs
4. Experiment with different text splitting strategies

### Batch Processing

For large document collections:

1. Organize documents into folders by topic or category
2. Create separate vector databases for each category
3. Use the appropriate vector database for domain-specific queries