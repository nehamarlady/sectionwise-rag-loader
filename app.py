from bs4 import BeautifulSoup
import sys
import os
import warnings
from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate

# Suppress all warnings
warnings.filterwarnings("ignore")

# CUSTOM LOADER: SectionAwareURLLoader
class SectionAwareURLLoader(BaseLoader):
    """
    A custom document loader that splits web pages into sections based on headings.
    
    This loader extends LangChain's BaseLoader and uses RecursiveUrlLoader internally
    to crawl documentation pages. It then splits each page into separate Document
    objects based on section headings (h1, h2, h3), providing better granularity
    for retrieval in RAG applications.
    
    Each section becomes a separate Document with metadata including:
    - source_url: The original page URL
    - section: The section heading/title
    
    If a page has no detectable headings, it returns the entire page as a single document.
    """
    
    def __init__(self, url: str, max_depth: int = 3):
        """
        Initialize the SectionAwareURLLoader.
        
        Args:
            url: The base URL to start crawling from
            max_depth: Maximum depth for recursive crawling (default: 3)
        """
        self.url = url
        self.max_depth = max_depth
    
    def load(self) -> List[Document]:
        """
        Load documents from the URL and split them into sections.
        
        Returns:
            List[Document]: A list of Document objects, one per section
        """
        # Use RecursiveUrlLoader internally to crawl pages
        internal_loader = RecursiveUrlLoader(
            url=self.url,
            max_depth=self.max_depth,
            extractor=lambda x: x  # Get raw HTML
        )
        
        pages = internal_loader.load()
        
        # Split each page into sections
        section_documents = []
        
        for page in pages:
            sections = self._split_page_into_sections(page)
            section_documents.extend(sections)
        
        return section_documents
    
    def _split_page_into_sections(self, page: Document) -> List[Document]:
        """
        Split a single page into multiple sections based on headings.
        
        Args:
            page: A Document object containing the full page HTML
            
        Returns:
            List[Document]: Multiple Document objects, one per section
        """
        # Parse the HTML
        soup = BeautifulSoup(page.page_content, 'html.parser')
        
        # Remove navigation, footer, and sidebar elements for cleaner content
        for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Find all section headings (h1, h2, h3)
        headings = soup.find_all(['h1', 'h2', 'h3'])
        
        # If no headings found, return the whole page as one document
        if not headings:
            return [Document(
                page_content=soup.get_text(separator='\n', strip=True),
                metadata={
                    "source_url": page.metadata.get("source", ""),
                    "section": "Full Page"
                }
            )]
        
        sections = []
        
        for i, heading in enumerate(headings):
            # Get section title (remove anchor links like ¶)
            section_title = heading.get_text(strip=True)
            # Clean up anchor symbols
            section_title = section_title.replace('¶', '').strip()
            
            # Collect content between this heading and the next
            content_parts = []
            current = heading.next_sibling
            
            # Get the next heading to know where to stop
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            while current:
                # Stop if we've reached the next heading
                if current == next_heading:
                    break
                
                # Stop if we hit another heading at same or higher level
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3']:
                    break
                
                # Extract text content from elements
                if hasattr(current, 'get_text'):
                    text = current.get_text(separator='\n', strip=True)
                    if text:
                        content_parts.append(text)
                elif isinstance(current, str):
                    text = current.strip()
                    if text:
                        content_parts.append(text)
                
                current = current.next_sibling
            
            # Combine section title and content
            section_content = '\n\n'.join(content_parts)
            section_text = f"{section_title}\n\n{section_content}" if section_content else section_title
            
            # Create Document for this section (only if it has content)
            if section_text.strip():
                sections.append(Document(
                    page_content=section_text,
                    metadata={
                        "source_url": page.metadata.get("source", ""),
                        "section": section_title
                    }
                ))
        
        # Return sections, or original page if no valid sections were created
        return sections if sections else [page]


# MAIN APPLICATION

print("Welcome to Python Tutorial Learning Assistant")
print("Loading Python tutorial documentation...")
print("This will take a few minutes to scrape and process the docs.")

# Read environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-pro")

# Python tutorial URL
url = 'https://docs.python.org/3/tutorial/index.html'

# Initialize the vector store
vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query"),
    persist_directory=".chromadb_python"
)

def load_docs(docs):
    """
    Loads documents into a vector store for efficient processing in batches.

    Args:
      docs (list): A list of documents, where each document is a string.

    Raises:
      ValueError: If the input `docs` is not a list.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Add documents in batches to avoid exceeding max batch size
    BATCH_SIZE = 5000  # Set below the max of 5461 to be safe
    total_splits = len(splits)
    num_batches = (total_splits + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"Total chunks to load: {total_splits}")
    print(f"Loading in {num_batches} batch(es)...\n")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_splits)
        batch = splits[start_idx:end_idx]
        
        vectorstore.add_documents(documents=batch)
        print(f"Loaded batch {batch_num + 1}/{num_batches}: {len(batch)} chunks ({end_idx}/{total_splits} total)")
    
    print(f"\nSuccessfully loaded all {total_splits} document chunks into the vector store.")


# Check if database already has data
existing_count = vectorstore._collection.count()

if existing_count == 0:
    print("Database is empty. Loading documents...")
    print(f'Crawling Python tutorial from {url}')
    print("Please wait while documents are being loaded and processed...")
    
    # Use the custom SectionAwareURLLoader 
    loader = SectionAwareURLLoader(
        url=url,
        max_depth=3
    )
    
    try:
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} sections from Python tutorial")
        
        load_docs(docs)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)
else:
    print(f"Database already loaded with {existing_count} chunks!")
    print("Using existing embeddings (instant load).")

# Create retriever (works whether we loaded fresh or used cache)
retriever = vectorstore.as_retriever()

llm = GoogleGenerativeAI(model=GOOGLE_MODEL)

def format_docs(docs):
    """Formats a list of documents into a single string with double newlines between documents.

    Args:
        docs: A list of dictionaries, where each dictionary represents a document
                and has a "page_content" key containing the document text.

    Returns:
        A string containing the formatted document content.
    """
    formatted_text = "\n\n".join(doc.page_content for doc in docs)
    return formatted_text


# HyDE document generation
template = """Please write a passage to answer the question about Python programming
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser() 
)

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever 


template = """I am learning Python. Help me understand the concept step by step with examples when relevant. 
Answer the following question based on this context from the official Python tutorial. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*60)
print("Python Tutorial RAG Assistant Ready!")
print("="*60)
print("\nDocuments loaded from:")

# Show which documents were loaded
document_data_sources = set()
for doc_metadata in retriever.vectorstore.get()['metadatas']:
    document_data_sources.add(doc_metadata.get('source_url', doc_metadata.get('source', 'Unknown')))
for doc in sorted(document_data_sources):
    print(f"  • {doc}")

print("\n" + "="*60)
print("Ask me anything about Python!")
print("Examples:")
print("  - How do I use list comprehensions?")
print("  - What's the difference between a list and a tuple?")
print("\nPress Enter (empty line) to Exit")
print("="*60 + "\n")

while True:
    line = input("python>> ")
    if line:
        try:
            retrieved_docs = retrieval_chain.invoke({"question": line})
            result = final_rag_chain.invoke({"context": format_docs(retrieved_docs), "question": line})
            print(f"\n{result}\n")
            
            # Show which sections were used to answer (NEW!)
            print("─" * 60)
            print("Answer based on these sections:")
            for i, doc in enumerate(retrieved_docs[:3], 1):  # Show top 3 sources
                section = doc.metadata.get('section', 'Unknown')
                source = doc.metadata.get('source_url', doc.metadata.get('source', 'Unknown'))
                print(f"  {i}. Section: '{section}'")
                print(f"     From: {source}")
            print("─" * 60 + "\n")
            
        except Exception as e:
            print(f"Error processing question: {e}\n")
    else:
        print("Goodbye! Happy Python learning!")
        break