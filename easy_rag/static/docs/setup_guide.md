# Easy RAG System - Setup Guide

## Introduction

This guide will walk you through the process of setting up the Easy RAG System on your computer. The Easy RAG System is designed to be user-friendly and requires minimal technical knowledge to set up and use.

## System Requirements

Before you begin, ensure your system meets the following requirements:

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB (16GB+ recommended for larger documents or models)
- **Disk Space**: At least 10GB of free space (more if using local LLMs)
- **Internet Connection**: Required for downloading dependencies and using API-based models

## Installation Steps

### Step 1: Install Python

If you don't already have Python installed, download and install it from the [official Python website](https://www.python.org/downloads/). Make sure to check the option to "Add Python to PATH" during installation.

To verify your Python installation, open a terminal or command prompt and run:

```bash
python --version
```

You should see the Python version number displayed.

### Step 2: Download the Easy RAG System

Download the Easy RAG System from the [official repository](https://github.com/example/easy-rag-system) (replace with actual repository URL).

You can either:
- Download the ZIP file and extract it to a folder of your choice
- Clone the repository using Git:

```bash
git clone https://github.com/example/easy-rag-system.git
cd easy-rag-system
```

### Step 3: Create a Virtual Environment (Recommended)

It's recommended to create a virtual environment to avoid conflicts with other Python packages:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

This may take a few minutes as it installs all the necessary packages.

### Step 5: Initialize the Database

Initialize the database by running:

```bash
python init_db.py
```

### Step 6: Configure Environment Variables (Optional)

If you plan to use API-based models or services, create a `.env` file in the root directory and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

You can also configure these later through the application's settings interface.

### Step 7: Start the Application

Start the Easy RAG System by running:

```bash
python app.py
```

The application will start and be accessible at `http://127.0.0.1:5000` in your web browser.

## First-Time Setup

When you first access the Easy RAG System, you'll be guided through a setup process:

1. **Document Storage Path**: Choose where uploaded documents will be stored
2. **Vector Database Path**: Choose where vector databases will be stored
3. **API Keys**: Optionally provide API keys for external services

You can change these settings later through the Settings page.

## Verifying Installation

To verify that everything is working correctly:

1. Navigate to the Dependencies page
2. Check that all core dependencies are installed
3. Upload a small text document to test document management
4. Create a simple vector database to test the core functionality

## Troubleshooting

If you encounter issues during installation:

- **Missing Dependencies**: Try installing them manually using `pip install package_name`
- **Database Errors**: Delete the database file and run `python init_db.py` again
- **Permission Errors**: Ensure you have write permissions for the application directory
- **Port Already in Use**: Change the port by modifying the `app.py` file

For more detailed troubleshooting, refer to the Troubleshooting section in the documentation.

## Next Steps

Now that you have successfully installed the Easy RAG System, you can:

- Upload your documents
- Create vector databases
- Configure retrievers
- Start querying your documents

Refer to the Usage Tutorials for detailed instructions on how to use these features.