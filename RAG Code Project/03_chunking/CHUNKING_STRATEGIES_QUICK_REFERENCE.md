# LlamaIndex Chunking Strategies - Quick Reference Guide

## 📋 All 16 Chunking Strategies at a Glance

### ✅ LlamaIndex Native Strategies (1-9)

#### 1. Sentence Splitter ⭐ MOST COMMON
```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" ",
    paragraph_separator="\n\n\n"
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** General text, articles, books  
**Pros:** Fast, respects sentence boundaries, configurable  
**Cons:** Fixed size, doesn't consider semantic meaning

---

#### 2. Sentence Window
```python
from llama_index.core.node_parser import SentenceWindowNodeParser

splitter = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Q&A systems, RAG applications  
**Pros:** Maintains context, great for retrieval  
**Cons:** More complex, larger storage

---

#### 3. Semantic Splitter 🧠
```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Topic-coherent documents  
**Pros:** AI-powered, semantically coherent chunks  
**Cons:** Slower, requires embeddings

---

#### 4. Token Text Splitter
```python
from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" "
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Precise token control for LLM limits  
**Pros:** Exact token counts, predictable  
**Cons:** May split mid-sentence

---

#### 5. Code Splitter 💻
```python
from llama_index.core.node_parser import CodeSplitter

splitter = CodeSplitter(
    language="python",  # python, javascript, java, cpp, go, rust, etc.
    chunk_lines=40,
    chunk_lines_overlap=5,
    max_chars=1500
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Source code files  
**Pros:** Respects code structure, language-aware  
**Cons:** Language-specific

**Supported languages:** Python, JavaScript, TypeScript, Java, C++, Go, Rust, PHP, Ruby, Swift, Kotlin

---

#### 6. Markdown Splitter
```python
from llama_index.core.node_parser import MarkdownNodeParser

splitter = MarkdownNodeParser()
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Markdown documentation, README files  
**Pros:** Preserves hierarchy, keeps sections together  
**Cons:** Only for markdown files

---

#### 7. HTML Splitter
```python
from llama_index.core.node_parser import HTMLNodeParser

splitter = HTMLNodeParser(
    tags=["p", "h1", "h2", "h3", "h4", "h5"]
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Web pages, HTML documentation  
**Pros:** Respects HTML structure  
**Cons:** Only for HTML files

---

#### 8. Hierarchical Splitter 🗂️
```python
from llama_index.core.node_parser import HierarchicalNodeParser

splitter = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Parent -> child sizes
)
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** Long documents, multi-level context  
**Pros:** Parent-child relationships, flexible retrieval  
**Cons:** More complex structure

---

#### 9. JSON Splitter
```python
from llama_index.core.node_parser import JSONNodeParser

splitter = JSONNodeParser()
nodes = splitter.get_nodes_from_documents(documents)
```
**Best for:** JSON data, API responses  
**Pros:** Preserves JSON structure  
**Cons:** Only for JSON files

---

### 🔗 Langchain Integration Strategies (10-16)

#### 10. Recursive Character Text Splitter ⭐ LANGCHAIN FAVORITE
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** General text, versatile use  
**Pros:** Tries multiple separators, very popular, industry standard  
**Cons:** Langchain dependency

**Why it's popular:** Intelligently tries splitting by paragraphs first, then sentences, then words, then characters - very robust!

---

#### 11. Character Text Splitter
```python
from langchain.text_splitter import CharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator="\n"
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Simple character-based control  
**Pros:** Simple, predictable  
**Cons:** Less intelligent than recursive

---

#### 12. SpaCy Text Splitter
```python
from langchain.text_splitter import SpacyTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = SpacyTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Linguistically-aware splitting  
**Pros:** NLP-powered, accurate sentences  
**Cons:** Requires spaCy installation

**Installation:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

#### 13. NLTK Text Splitter
```python
from langchain.text_splitter import NLTKTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = NLTKTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Academic text, complex sentences  
**Pros:** Punkt tokenizer, good for formal text  
**Cons:** Requires NLTK installation

**Installation:**
```bash
pip install nltk
# Then in Python: import nltk; nltk.download('punkt')
```

---

#### 14. Markdown Header Text Splitter
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]
langchain_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Markdown with header metadata  
**Pros:** Preserves header hierarchy as metadata  
**Cons:** Only for markdown

---

#### 15. Python Code Splitter
```python
from langchain.text_splitter import PythonCodeTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = PythonCodeTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Python source code  
**Pros:** Python-aware splitting  
**Cons:** Python only (use LlamaIndex #5 for multi-language)

---

#### 16. LaTeX Text Splitter
```python
from langchain.text_splitter import LatexTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

langchain_splitter = LatexTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
parser = LangchainNodeParser(langchain_splitter)
nodes = parser.get_nodes_from_documents(documents)
```
**Best for:** Academic papers, scientific documents  
**Pros:** LaTeX-aware, preserves mathematical formatting  
**Cons:** Only for LaTeX files

---

## 🎯 Decision Tree: Which Splitter Should I Use?

```
Start Here
│
├─ Working with CODE?
│  ├─ YES → Multi-language? → #5 Code Splitter (LlamaIndex)
│  └─ Python only? → #15 Python Code (Langchain)
│
├─ Building Q&A/RAG system?
│  ├─ Need context windows? → #2 Sentence Window
│  └─ Need semantic coherence? → #3 Semantic Splitter
│
├─ Academic/Scientific papers?
│  ├─ LaTeX format? → #16 LaTeX Splitter
│  └─ PDF/Other? → #13 NLTK Text Splitter
│
├─ Markdown documentation?
│  ├─ Need header metadata? → #14 Markdown Header (LC)
│  └─ Simple splitting? → #6 Markdown Splitter
│
├─ Web pages/HTML?
│  └─ #7 HTML Splitter
│
├─ JSON/API data?
│  └─ #9 JSON Splitter
│
├─ Long documents needing hierarchy?
│  └─ #8 Hierarchical Splitter
│
├─ Already using Langchain?
│  └─ #10 Recursive Character (LC) ⭐
│
└─ General text/books?
   ├─ Want industry standard? → #10 Recursive Character (LC)
   └─ Pure LlamaIndex? → #1 Sentence Splitter
```

---

## 📊 Performance Comparison

| Strategy | Speed | Intelligence | Context Awareness | Best Use Case |
|----------|-------|--------------|-------------------|---------------|
| #1 Sentence | ⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐ | General text |
| #2 Window | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Q&A systems |
| #3 Semantic | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Topic coherence |
| #4 Token | ⚡⚡⚡⚡ | ⭐⭐ | ⭐ | Token limits |
| #5 Code | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Source code |
| #10 Recursive | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Versatile |

---

## 🚀 Installation Commands

### Minimal (LlamaIndex Native)
```bash
pip install llama-index llama-index-core
pip install llama-index-embeddings-openai llama-index-llms-openai
```

### Full (All Strategies)
```bash
# Core LlamaIndex
pip install llama-index llama-index-core
pip install llama-index-embeddings-openai llama-index-llms-openai

# Document readers
pip install pypdf docx2txt python-pptx pandas openpyxl beautifulsoup4

# Langchain integration
pip install langchain langchain-text-splitters

# Optional NLP
pip install spacy nltk
python -m spacy download en_core_web_sm
```

---

## 💡 Pro Tips

1. **Start with #10 Recursive Character** if you're unsure - it's the most versatile
2. **Use #3 Semantic Splitter** for maximum quality (but it's slower)
3. **Use #2 Sentence Window** for RAG systems - the context helps a lot
4. **Always set chunk_overlap** to prevent losing context at boundaries
5. **For code**, use language-specific splitters (#5 or #15)
6. **Test different strategies** - results vary by document type!

---

## 📝 Complete Example

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Choose your splitter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# Create chunks
nodes = splitter.get_nodes_from_documents(documents)

# Create index
index = VectorStoreIndex(nodes)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Your question here")
print(response)
```

---

## 🔗 Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Complete Code Example](./llamaindex_complete_chunking_demo.py)

---

**Last Updated:** February 2026  
**LlamaIndex Version:** Latest (0.10+)  
**Langchain Version:** Latest (0.1+)
