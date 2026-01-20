# SectionWiseRAG
### Section-Aware Web Document Loader for Retrieval-Augmented Generation (RAG)

## Overview
SectionWiseRAG is a custom document loader built on top of **LangChain** that crawls web documentation pages and splits them into **section-level documents** using HTML headings (`h1`, `h2`, `h3`).

Unlike traditional loaders that treat an entire webpage as a single document, this project produces **granular, semantically meaningful chunks**, making it highly effective for **Retrieval-Augmented Generation (RAG)** pipelines.


## Motivation
Most web loaders:
- Load entire pages as one large document
- Ignore semantic structure
- Reduce retrieval accuracy in vector databases

SectionWiseRAG improves **retrieval precision and LLM response quality** by preserving the logical structure of documentation.



## Features
- ✅ Section-aware splitting using HTML headings
- ✅ Recursive web crawling with depth control
- ✅ Metadata-rich LangChain `Document` objects
- ✅ Fallback handling for pages without headings
- ✅ Easy integration with vector stores and LLMs



## Architecture
Web URL -> RecursiveUrlLoader -> BeautifulSoup Parsing -> Heading-Based Section Splitting -> LangChain Documents -> Vector Store (Chroma) -> LLM Retrieval & QA

## Tech Stack
- Python
- LangChain
- BeautifulSoup4
- ChromaDB
- Google Gemini (langchain-google-genai)
- python-dotenv
- 
## Installation

```bash
git clone https://github.com/your-username/sectionwise-rag-loader.git
cd sectionwise-rag-loader
pip install -r requirements.txt
GOOGLE_API_KEY=your_api_key_here
python app.py
```

#### Each HTML section is converted into an individual LangChain Document with metadata:
- source_url
- section_title
