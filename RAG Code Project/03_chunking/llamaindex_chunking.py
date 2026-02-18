"""
LlamaIndex Complete Chunking & Document Loading Demo
====================================================
This file demonstrates ALL chunking strategies and document loaders in LlamaIndex (latest version).
Each section is clearly marked with explanations.

Installation required:
pip install llama-index llama-index-core llama-index-embeddings-openai llama-index-llms-openai
pip install pypdf docx2txt python-pptx ebooklib beautifulsoup4 pandas openpyxl

# For Langchain integration (Strategies 10-16):
pip install langchain langchain-text-splitters

# Optional NLP libraries:
pip install spacy nltk
python -m spacy download en_core_web_sm
# In Python: import nltk; nltk.download('punkt')
"""

import os
from pathlib import Path
from typing import List

# Core LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.core.node_parser import (
    # Sentence-based splitters
    SentenceSplitter,
    SentenceWindowNodeParser,
    
    # Semantic splitters
    SemanticSplitterNodeParser,
    
    # Token-based splitters
    TokenTextSplitter,
    
    # Code splitters
    CodeSplitter,
    
    # Markdown splitters
    MarkdownNodeParser,
    
    # HTML splitters
    HTMLNodeParser,
    
    # Hierarchical splitters
    HierarchicalNodeParser,
    
    # JSON splitters
    JSONNodeParser,
)

# Embedding models
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# =============================================================================
# SECTION 1: SETUP AND CONFIGURATION
# =============================================================================

def setup_llamaindex():
    """Configure LlamaIndex with OpenAI (you can use other LLMs/embeddings)"""
    Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    print("✓ LlamaIndex configured successfully\n")


# =============================================================================
# SECTION 2: DOCUMENT LOADING - ALL FILE TYPES
# =============================================================================

def load_all_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Load ALL document types from a folder using SimpleDirectoryReader.
    
    Supported formats:
    - Text: .txt, .md, .csv, .json
    - PDFs: .pdf
    - Word: .doc, .docx
    - PowerPoint: .ppt, .pptx
    - Excel: .xlsx, .xls
    - Images: .jpg, .png (with OCR if configured)
    - HTML: .html, .htm
    - EPub: .epub
    - And many more...
    """
    print(f"📂 Loading documents from: {folder_path}")
    
    # SimpleDirectoryReader automatically detects file types
    reader = SimpleDirectoryReader(
        input_dir=folder_path,
        recursive=True,  # Search subdirectories
        required_exts=None,  # Accept all file types
        exclude_hidden=True,  # Skip hidden files
        errors='ignore'  # Continue on errors
    )
    
    documents = reader.load_data()
    print(f"✓ Loaded {len(documents)} documents\n")
    return documents


def load_specific_file_types(folder_path: str):
    """Load specific file types with custom configurations"""
    
    # PDF files only
    pdf_reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".pdf"],
        recursive=True
    )
    pdf_docs = pdf_reader.load_data()
    print(f"📄 PDF documents: {len(pdf_docs)}")
    
    # Word documents only
    docx_reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".docx", ".doc"],
        recursive=True
    )
    docx_docs = docx_reader.load_data()
    print(f"📝 Word documents: {len(docx_docs)}")
    
    # Excel files only
    excel_reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".xlsx", ".xls"],
        recursive=True
    )
    excel_docs = excel_reader.load_data()
    print(f"📊 Excel documents: {len(excel_docs)}")
    
    # Markdown files only
    md_reader = SimpleDirectoryReader(
        input_dir=folder_path,
        required_exts=[".md"],
        recursive=True
    )
    md_docs = md_reader.load_data()
    print(f"📋 Markdown documents: {len(md_docs)}")
    
    return pdf_docs + docx_docs + excel_docs + md_docs


# =============================================================================
# SECTION 3: CHUNKING STRATEGY #1 - SENTENCE SPLITTER (Most Common)
# =============================================================================

def chunking_sentence_splitter(documents: List[Document]):
    """
    SENTENCE SPLITTER: Splits text by sentences with overlap.
    
    Best for: General text, articles, books
    - Respects sentence boundaries
    - Configurable chunk size and overlap
    - Most commonly used splitter
    """
    print("🔪 CHUNKING STRATEGY 1: Sentence Splitter")
    
    splitter = SentenceSplitter(
        chunk_size=512,        # Target chunk size in tokens
        chunk_overlap=50,      # Overlap between chunks
        separator=" ",         # Separator for splitting
        paragraph_separator="\n\n\n",
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} chunks")
    print(f"   ✓ Sample chunk: {nodes[0].text[:100]}...\n")
    return nodes


# =============================================================================
# SECTION 4: CHUNKING STRATEGY #2 - SENTENCE WINDOW
# =============================================================================

def chunking_sentence_window(documents: List[Document]):
    """
    SENTENCE WINDOW: Creates chunks with surrounding context window.
    
    Best for: Q&A systems where context matters
    - Each node is a sentence
    - Stores surrounding sentences as metadata
    - Allows retrieval of small chunks with large context
    """
    print("🔪 CHUNKING STRATEGY 2: Sentence Window")
    
    splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3,          # Number of sentences on each side
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} sentence nodes with context windows")
    print(f"   ✓ Each node has ±3 sentences of context\n")
    return nodes


# =============================================================================
# SECTION 5: CHUNKING STRATEGY #3 - SEMANTIC SPLITTER
# =============================================================================

def chunking_semantic_splitter(documents: List[Document]):
    """
    SEMANTIC SPLITTER: Splits based on semantic similarity.
    
    Best for: Documents where topic coherence is important
    - Uses embeddings to find natural breakpoints
    - Chunks stay semantically coherent
    - More intelligent than fixed-size splitting
    """
    print("🔪 CHUNKING STRATEGY 3: Semantic Splitter")
    
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,                    # Number of sentences to group
        breakpoint_percentile_threshold=95,  # Threshold for splitting
        embed_model=Settings.embed_model,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} semantically coherent chunks")
    print(f"   ✓ Chunks split at natural topic boundaries\n")
    return nodes


# =============================================================================
# SECTION 6: CHUNKING STRATEGY #4 - TOKEN TEXT SPLITTER
# =============================================================================

def chunking_token_splitter(documents: List[Document]):
    """
    TOKEN TEXT SPLITTER: Splits text based on token count.
    
    Best for: When you need precise token control for LLM context
    - Counts tokens using tokenizer
    - Ensures chunks fit within token limits
    - Good for LLM API calls with token limits
    """
    print("🔪 CHUNKING STRATEGY 4: Token Text Splitter")
    
    splitter = TokenTextSplitter(
        chunk_size=512,        # Maximum tokens per chunk
        chunk_overlap=50,      # Overlap in tokens
        separator=" ",
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} token-based chunks")
    print(f"   ✓ Each chunk ≤ 512 tokens\n")
    return nodes


# =============================================================================
# SECTION 7: CHUNKING STRATEGY #5 - CODE SPLITTER
# =============================================================================

def chunking_code_splitter(documents: List[Document], language: str = "python"):
    """
    CODE SPLITTER: Specialized for source code files.
    
    Best for: Programming code, scripts
    - Respects code structure (functions, classes)
    - Language-aware splitting
    - Supports: Python, JavaScript, TypeScript, Java, C++, Go, Rust, etc.
    """
    print(f"🔪 CHUNKING STRATEGY 5: Code Splitter ({language})")
    
    splitter = CodeSplitter(
        language=language,     # Programming language
        chunk_lines=40,        # Target lines per chunk
        chunk_lines_overlap=5, # Overlap in lines
        max_chars=1500,        # Maximum characters
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} code chunks")
    print(f"   ✓ Respects {language} syntax structure\n")
    return nodes


# =============================================================================
# SECTION 8: CHUNKING STRATEGY #6 - MARKDOWN SPLITTER
# =============================================================================

def chunking_markdown_splitter(documents: List[Document]):
    """
    MARKDOWN SPLITTER: Splits markdown by headers and structure.
    
    Best for: Markdown documentation, README files
    - Respects markdown headers (h1, h2, h3, etc.)
    - Preserves document hierarchy
    - Keeps related sections together
    """
    print("🔪 CHUNKING STRATEGY 6: Markdown Splitter")
    
    splitter = MarkdownNodeParser()
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} markdown chunks")
    print(f"   ✓ Split by headers and structure\n")
    return nodes


# =============================================================================
# SECTION 9: CHUNKING STRATEGY #7 - HTML SPLITTER
# =============================================================================

def chunking_html_splitter(documents: List[Document]):
    """
    HTML SPLITTER: Splits HTML documents by tags.
    
    Best for: Web pages, HTML documentation
    - Respects HTML structure
    - Splits by tags (div, p, section, etc.)
    - Preserves important HTML metadata
    """
    print("🔪 CHUNKING STRATEGY 7: HTML Splitter")
    
    splitter = HTMLNodeParser(
        tags=["p", "h1", "h2", "h3", "h4", "h5"],  # Tags to split on
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} HTML chunks")
    print(f"   ✓ Split by HTML tags\n")
    return nodes


# =============================================================================
# SECTION 10: CHUNKING STRATEGY #8 - HIERARCHICAL SPLITTER
# =============================================================================

def chunking_hierarchical_splitter(documents: List[Document]):
    """
    HIERARCHICAL SPLITTER: Creates parent-child chunk relationships.
    
    Best for: Long documents requiring multi-level context
    - Creates larger parent chunks
    - Smaller child chunks for retrieval
    - Maintains hierarchical relationships
    """
    print("🔪 CHUNKING STRATEGY 8: Hierarchical Splitter")
    
    splitter = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],  # Parent -> child sizes
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} hierarchical chunks")
    print(f"   ✓ 3 levels: 2048 → 512 → 128 tokens\n")
    return nodes


# =============================================================================
# SECTION 11: CHUNKING STRATEGY #9 - JSON SPLITTER
# =============================================================================

def chunking_json_splitter(documents: List[Document]):
    """
    JSON SPLITTER: Splits JSON documents intelligently.
    
    Best for: JSON data, API responses, structured data
    - Preserves JSON structure
    - Splits by JSON objects/arrays
    - Maintains data relationships
    """
    print("🔪 CHUNKING STRATEGY 9: JSON Splitter")
    
    splitter = JSONNodeParser()
    
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   ✓ Created {len(nodes)} JSON chunks")
    print(f"   ✓ Respects JSON structure\n")
    return nodes


# =============================================================================
# SECTION 12: CHUNKING STRATEGY #10 - RECURSIVE CHARACTER TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_recursive_character_splitter(documents: List[Document]):
    """
    RECURSIVE CHARACTER TEXT SPLITTER: Uses Langchain's popular splitter.
    
    Best for: General text with intelligent splitting
    - Tries multiple separators recursively: \n\n, \n, space, character
    - Most commonly used in Langchain
    - Maintains natural text boundaries
    - Can be integrated with LlamaIndex via LangchainNodeParser
    """
    print("🔪 CHUNKING STRATEGY 10: Recursive Character Text Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        # Create Langchain's RecursiveCharacterTextSplitter
        langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,           # Maximum chunk size
            chunk_overlap=50,         # Overlap between chunks
            length_function=len,      # Function to measure chunk size
            separators=["\n\n", "\n", " ", ""],  # Try these separators in order
            is_separator_regex=False,
        )
        
        # Wrap it in LlamaIndex's LangchainNodeParser
        parser = LangchainNodeParser(langchain_splitter)
        
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} recursive chunks")
        print(f"   ✓ Uses Langchain's intelligent recursive splitting\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain not installed. Install: pip install langchain")
        print("   ⚠ Skipping this chunking strategy\n")
        return []


# =============================================================================
# SECTION 13: CHUNKING STRATEGY #11 - CHARACTER TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_character_text_splitter(documents: List[Document]):
    """
    CHARACTER TEXT SPLITTER: Simple character-based splitting from Langchain.
    
    Best for: When you need exact character-based control
    - Splits by single separator (like newline)
    - Simple and predictable
    - Good for structured text with clear delimiters
    """
    print("🔪 CHUNKING STRATEGY 11: Character Text Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import CharacterTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        langchain_splitter = CharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separator="\n",          # Split on newlines
            length_function=len,
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} character-based chunks")
        print(f"   ✓ Split on newline separator\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain not installed\n")
        return []


# =============================================================================
# SECTION 14: CHUNKING STRATEGY #12 - SPACY TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_spacy_text_splitter(documents: List[Document]):
    """
    SPACY TEXT SPLITTER: Uses spaCy's NLP for intelligent splitting.
    
    Best for: Linguistically-aware splitting
    - Uses spaCy's sentence detection
    - More accurate sentence boundaries
    - Requires spaCy installation
    """
    print("🔪 CHUNKING STRATEGY 12: SpaCy Text Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import SpacyTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        langchain_splitter = SpacyTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separator="\n\n",
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} spaCy-based chunks")
        print(f"   ✓ Uses NLP for sentence detection\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain and/or spacy not installed")
        print("   ⚠ Install: pip install langchain spacy")
        print("   ⚠ Then: python -m spacy download en_core_web_sm\n")
        return []


# =============================================================================
# SECTION 15: CHUNKING STRATEGY #13 - NLTK TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_nltk_text_splitter(documents: List[Document]):
    """
    NLTK TEXT SPLITTER: Uses NLTK for sentence tokenization.
    
    Best for: Academic text, formal documents
    - Uses NLTK's punkt tokenizer
    - Good for complex sentence structures
    - Requires NLTK installation
    """
    print("🔪 CHUNKING STRATEGY 13: NLTK Text Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import NLTKTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        langchain_splitter = NLTKTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} NLTK-based chunks")
        print(f"   ✓ Uses NLTK sentence tokenizer\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain and/or nltk not installed")
        print("   ⚠ Install: pip install langchain nltk")
        print("   ⚠ Then run in Python: import nltk; nltk.download('punkt')\n")
        return []


# =============================================================================
# SECTION 16: CHUNKING STRATEGY #14 - MARKDOWN HEADER TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_langchain_markdown_splitter(documents: List[Document]):
    """
    MARKDOWN HEADER TEXT SPLITTER: Langchain's markdown splitter.
    
    Best for: Markdown files with clear header structure
    - Splits on markdown headers (# ## ### etc.)
    - Preserves header hierarchy as metadata
    - Keeps sections together
    """
    print("🔪 CHUNKING STRATEGY 14: Markdown Header Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        langchain_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} markdown header chunks")
        print(f"   ✓ Split by header hierarchy\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain not installed\n")
        return []


# =============================================================================
# SECTION 17: CHUNKING STRATEGY #15 - PYTHON CODE SPLITTER (Langchain)
# =============================================================================

def chunking_langchain_python_splitter(documents: List[Document]):
    """
    PYTHON CODE SPLITTER: Langchain's Python-specific code splitter.
    
    Best for: Python source code
    - Splits on Python syntax (functions, classes)
    - Preserves code structure
    - Similar to LlamaIndex CodeSplitter but from Langchain
    """
    print("🔪 CHUNKING STRATEGY 15: Python Code Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import PythonCodeTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        langchain_splitter = PythonCodeTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} Python code chunks")
        print(f"   ✓ Respects Python syntax\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain not installed\n")
        return []


# =============================================================================
# SECTION 18: CHUNKING STRATEGY #16 - LATEX TEXT SPLITTER (Langchain)
# =============================================================================

def chunking_latex_splitter(documents: List[Document]):
    """
    LATEX TEXT SPLITTER: Specialized for LaTeX documents.
    
    Best for: Academic papers, scientific documents in LaTeX
    - Splits on LaTeX structure (sections, subsections)
    - Preserves LaTeX formatting
    - Good for mathematical documents
    """
    print("🔪 CHUNKING STRATEGY 16: LaTeX Text Splitter (Langchain)")
    
    try:
        from langchain.text_splitter import LatexTextSplitter
        from llama_index.core.node_parser import LangchainNodeParser
        
        langchain_splitter = LatexTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
        
        parser = LangchainNodeParser(langchain_splitter)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"   ✓ Created {len(nodes)} LaTeX chunks")
        print(f"   ✓ Preserves LaTeX structure\n")
        return nodes
        
    except ImportError:
        print("   ⚠ langchain not installed\n")
        return []


# =============================================================================
# SECTION 19: CREATING INDEXES AND QUERYING
# =============================================================================

def create_and_query_index(nodes):
    """Create vector index and perform sample query"""
    print("🔍 Creating Vector Index...")
    
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine()
    
    print("✓ Index created successfully")
    print("✓ Ready for queries!\n")
    
    return index, query_engine


# =============================================================================
# SECTION 20: COMPARISON FUNCTION
# =============================================================================

def compare_all_chunking_strategies(documents: List[Document]):
    """
    Compare all chunking strategies side-by-side
    Shows chunk counts and characteristics for each method
    """
    print("=" * 80)
    print("CHUNKING STRATEGY COMPARISON - ALL 16 METHODS")
    print("=" * 80)
    
    strategies = {
        # LlamaIndex Native Splitters
        "1. Sentence Splitter": lambda: chunking_sentence_splitter(documents),
        "2. Sentence Window": lambda: chunking_sentence_window(documents),
        "3. Semantic Splitter": lambda: chunking_semantic_splitter(documents),
        "4. Token Splitter": lambda: chunking_token_splitter(documents),
        "5. Code Splitter": lambda: chunking_code_splitter(documents, "python"),
        "6. Markdown Splitter": lambda: chunking_markdown_splitter(documents),
        "7. HTML Splitter": lambda: chunking_html_splitter(documents),
        "8. Hierarchical": lambda: chunking_hierarchical_splitter(documents),
        "9. JSON Splitter": lambda: chunking_json_splitter(documents),
        
        # Langchain Integration Splitters
        "10. Recursive Character (LC)": lambda: chunking_recursive_character_splitter(documents),
        "11. Character Text (LC)": lambda: chunking_character_text_splitter(documents),
        "12. SpaCy Text (LC)": lambda: chunking_spacy_text_splitter(documents),
        "13. NLTK Text (LC)": lambda: chunking_nltk_text_splitter(documents),
        "14. Markdown Header (LC)": lambda: chunking_langchain_markdown_splitter(documents),
        "15. Python Code (LC)": lambda: chunking_langchain_python_splitter(documents),
        "16. LaTeX Text (LC)": lambda: chunking_latex_splitter(documents),
    }
    
    results = {}
    for name, strategy_func in strategies.items():
        try:
            nodes = strategy_func()
            results[name] = len(nodes) if nodes else "Skipped (dependencies missing)"
        except Exception as e:
            results[name] = f"Error: {str(e)[:40]}"
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 80)
    print(f"{'Strategy':<35} {'Chunks Created':<20}")
    print("-" * 80)
    for name, count in results.items():
        print(f"{name:<35} {str(count):<20}")
    print("=" * 80 + "\n")
    
    return results


# =============================================================================
# SECTION 21: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating all features"""
    
    print("\n" + "=" * 80)
    print("LLAMAINDEX COMPLETE CHUNKING & DOCUMENT LOADING DEMO")
    print("All 16 Chunking Strategies + All Document Types")
    print("=" * 80 + "\n")
    
    # Setup
    setup_llamaindex()
    
    # Create sample documents folder (you should replace with your folder)
    sample_folder = "/home/claude/sample_documents"
    
    # Example: Create sample documents for testing
    print("📝 Creating sample documents for demonstration...\n")
    os.makedirs(sample_folder, exist_ok=True)
    
    # Sample text document
    with open(f"{sample_folder}/sample.txt", "w") as f:
        f.write("This is a sample text document. " * 100)
    
    # Sample markdown
    with open(f"{sample_folder}/sample.md", "w") as f:
        f.write("# Header\n\n## Subheader\n\nContent here. " * 50)
    
    # Sample JSON
    with open(f"{sample_folder}/sample.json", "w") as f:
        f.write('{"key": "value", "data": ["item1", "item2", "item3"]}')
    
    # Sample Python code
    with open(f"{sample_folder}/sample.py", "w") as f:
        f.write("def hello():\n    print('Hello World')\n\n" * 20)
    
    # Sample HTML
    with open(f"{sample_folder}/sample.html", "w") as f:
        f.write("<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>" * 20)
    
    print("✓ Sample documents created\n")
    
    # STEP 1: Load all documents
    all_documents = load_all_documents_from_folder(sample_folder)
    
    # STEP 2: Demonstrate LLAMAINDEX native chunking strategies
    print("\n" + "=" * 80)
    print("DEMONSTRATING LLAMAINDEX NATIVE CHUNKING STRATEGIES (1-9)")
    print("=" * 80 + "\n")
    
    nodes_sentence = chunking_sentence_splitter(all_documents)
    nodes_window = chunking_sentence_window(all_documents)
    nodes_semantic = chunking_semantic_splitter(all_documents)
    nodes_token = chunking_token_splitter(all_documents)
    nodes_code = chunking_code_splitter(all_documents, "python")
    nodes_markdown = chunking_markdown_splitter(all_documents)
    nodes_html = chunking_html_splitter(all_documents)
    nodes_hierarchical = chunking_hierarchical_splitter(all_documents)
    nodes_json = chunking_json_splitter(all_documents)
    
    # STEP 3: Demonstrate LANGCHAIN integration strategies
    print("\n" + "=" * 80)
    print("DEMONSTRATING LANGCHAIN INTEGRATION STRATEGIES (10-16)")
    print("=" * 80 + "\n")
    
    nodes_recursive = chunking_recursive_character_splitter(all_documents)
    nodes_char = chunking_character_text_splitter(all_documents)
    nodes_spacy = chunking_spacy_text_splitter(all_documents)
    nodes_nltk = chunking_nltk_text_splitter(all_documents)
    nodes_md_header = chunking_langchain_markdown_splitter(all_documents)
    nodes_python = chunking_langchain_python_splitter(all_documents)
    nodes_latex = chunking_latex_splitter(all_documents)
    
    # STEP 4: Create index with one strategy (you can choose any)
    if nodes_sentence:
        index, query_engine = create_and_query_index(nodes_sentence)
    
    # STEP 5: Compare all strategies
    compare_all_chunking_strategies(all_documents)
    
    print("✅ Demo completed successfully!")
    print("\n" + "=" * 80)
    print("NOTES:")
    print("=" * 80)
    print("1. Replace 'sample_documents' folder with your actual data folder")
    print("2. Install langchain for strategies 10-16: pip install langchain")
    print("3. Install optional deps: pip install spacy nltk")
    print("4. Each chunking strategy has different use cases - choose wisely!")
    print("5. The RecursiveCharacterTextSplitter (#10) is MOST POPULAR in Langchain")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE GUIDE
# =============================================================================
"""
QUICK START GUIDE:
==================

1. Install dependencies:
   # Core LlamaIndex
   pip install llama-index llama-index-embeddings-openai llama-index-llms-openai
   pip install pypdf docx2txt python-pptx pandas openpyxl
   
   # For Langchain integration (Strategies 10-16)
   pip install langchain langchain-text-splitters
   
   # Optional NLP libraries
   pip install spacy nltk
   python -m spacy download en_core_web_sm

2. Set OpenAI API key:
   export OPENAI_API_KEY='your-key-here'

3. Run the script:
   python llamaindex_complete_chunking_demo.py

4. Replace sample_folder with your document folder path


CHOOSING THE RIGHT CHUNKING STRATEGY (All 16 Methods):
=======================================================

LLAMAINDEX NATIVE STRATEGIES (1-9):
-------------------------------------
1️⃣  Sentence Splitter → General text, articles, books (MOST COMMON)
2️⃣  Sentence Window → Q&A systems needing context windows
3️⃣  Semantic Splitter → Topic-coherent documents (uses AI)
4️⃣  Token Splitter → Precise token control for LLM limits
5️⃣  Code Splitter → Source code files (Python, JS, Java, etc.)
6️⃣  Markdown Splitter → Markdown documentation
7️⃣  HTML Splitter → Web pages, HTML files
8️⃣  Hierarchical → Long documents with multi-level context
9️⃣  JSON Splitter → JSON data, API responses

LANGCHAIN INTEGRATION STRATEGIES (10-16):
------------------------------------------
🔟 Recursive Character (LC) → MOST POPULAR! General text (tries multiple separators)
1️⃣1️⃣ Character Text (LC) → Simple character-based splitting
1️⃣2️⃣ SpaCy Text (LC) → NLP-powered sentence detection
1️⃣3️⃣ NLTK Text (LC) → Academic text, complex sentences
1️⃣4️⃣ Markdown Header (LC) → Markdown with header metadata
1️⃣5️⃣ Python Code (LC) → Python source code
1️⃣6️⃣ LaTeX Text (LC) → Academic papers, scientific documents


WHEN TO USE LANGCHAIN vs LLAMAINDEX SPLITTERS:
===============================================

USE LANGCHAIN INTEGRATION (10-16) WHEN:
- You're already using Langchain in your project
- You need RecursiveCharacterTextSplitter (industry standard)
- You need NLP-powered splitting (SpaCy, NLTK)
- You're working with LaTeX documents
- You want header metadata from Markdown

USE LLAMAINDEX NATIVE (1-9) WHEN:
- Pure LlamaIndex project (better integration)
- You need semantic splitting (AI-powered)
- You need sentence windows for RAG
- You need hierarchical parent-child relationships
- You're working with HTML or general code


MOST POPULAR CHOICES BY USE CASE:
==================================
📚 General Text/Books → #1 Sentence Splitter OR #10 Recursive Character (LC)
💬 Q&A/RAG Systems → #2 Sentence Window OR #3 Semantic Splitter
💻 Source Code → #5 Code Splitter OR #15 Python Code (LC)
📝 Markdown Docs → #6 Markdown Splitter OR #14 Markdown Header (LC)
🌐 Web Pages → #7 HTML Splitter
📊 JSON/API Data → #9 JSON Splitter
🔬 Academic Papers → #13 NLTK Text (LC) OR #16 LaTeX (LC)
📖 Long Documents → #8 Hierarchical Splitter


SUPPORTED FILE TYPES:
======================
✅ Text: .txt, .md, .csv
✅ PDFs: .pdf
✅ Word: .doc, .docx
✅ PowerPoint: .ppt, .pptx
✅ Excel: .xlsx, .xls
✅ Code: .py, .js, .java, .cpp, .go, .rs, .ts, etc.
✅ Web: .html, .htm
✅ Data: .json, .xml
✅ eBooks: .epub
✅ Images: .jpg, .png (with OCR)
✅ LaTeX: .tex


REAL-WORLD EXAMPLES:
====================

Example 1: Building a chatbot for company documentation
→ Use #2 Sentence Window for better context retrieval

Example 2: Code search engine
→ Use #5 Code Splitter (LlamaIndex) or #15 Python Code (Langchain)

Example 3: Research paper Q&A system
→ Use #16 LaTeX Splitter for academic papers

Example 4: General knowledge base from mixed documents
→ Use #10 Recursive Character (most versatile)

Example 5: Legal document analysis
→ Use #3 Semantic Splitter to keep coherent sections together


PERFORMANCE TIPS:
=================
🚀 Fastest: #1 Sentence Splitter, #4 Token Splitter
🧠 Smartest: #3 Semantic Splitter (requires embeddings)
⚖️ Balanced: #10 Recursive Character Splitter
📏 Most Control: #4 Token Splitter, #11 Character Text
🔗 Best Context: #2 Sentence Window, #8 Hierarchical
"""