# Gaming RAG Research Agent 🎮

An intelligent AI-powered research agent designed for the video game industry. This project demonstrates how to build sophisticated AI agents that combine retrieval-augmented generation (RAG) with web search capabilities to answer complex questions about video games.

## 🌟 Features

- **Intelligent Game Research**: Query a comprehensive database of video games with natural language
- **Hybrid Knowledge System**: Combines local vector database search with real-time web search
- **Conversational AI**: Maintains context across conversations with memory management
- **Structured Outputs**: Returns well-formatted, actionable responses
- **Extensible Architecture**: Modular design for easy customization and extension

## 🏗️ Architecture

The project is built around two core components:

### 1. RAG (Retrieval-Augmented Generation) System
- **Vector Database**: ChromaDB for efficient semantic search of game information
- **Embedding-Based Retrieval**: Finds relevant games based on semantic similarity
- **Game Knowledge Base**: 15 carefully curated games with detailed metadata

### 2. Intelligent Agent Framework
- **State Management**: Tracks conversation context and user queries
- **Tool Integration**: Seamlessly combines multiple information sources
- **Memory System**: Maintains short-term and long-term memory for conversations
- **Evaluation Framework**: Assesses retrieval quality and response accuracy

## 📊 Game Dataset

The knowledge base includes diverse games across genres and platforms:

| Game | Platform | Genre | Year | Publisher |
|------|----------|-------|------|-----------|
| Gran Turismo | PlayStation 1 | Racing | 1997 | Sony Computer Entertainment |
| Super Mario Bros. | NES | Platformer | 1985 | Nintendo |
| The Witcher 3 | PC/Console | RPG | 2015 | CD Projekt |
| *...and 12 more* | | | | |

Each game entry contains:
- **Name**: Game title
- **Platform**: Gaming system(s)
- **Genre**: Game category
- **Publisher**: Publishing company
- **Description**: Detailed game overview
- **Year of Release**: Publication year

## 🛠️ Core Components

### Agent System (`lib/agents.py`)
- Conversational AI agent with tool integration
- State management for multi-turn conversations
- Token tracking and session management

### Vector Database (`lib/vector_db.py`)
- ChromaDB integration for semantic search
- Embedding-based document retrieval
- Persistent storage and indexing

### Tools & Utilities (`lib/tooling.py`)
- `retrieve_game`: Search local game database
- `evaluate_retrieval`: Assess search result quality
- `game_web_search`: Perform web searches for additional information

### Memory Management (`lib/memory.py`)
- Short-term conversation memory
- Context preservation across interactions
- Efficient memory cleanup and management

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- OpenAI API key
- Tavily API key (for web search)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gaming-rag-research-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install uv
   uv sync
   ```

3. **Environment Setup**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY="your_openai_api_key"
   CHROMA_OPENAI_API_KEY="your_openai_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

### Quick Start

1. **Explore with Jupyter notebooks**:
   - `Udaplay_01_starter_project.ipynb`: RAG system implementation
   - `Udaplay_02_starter_project.ipynb`: Complete agent development

### Running Tests

Execute the test suite to verify functionality:
```bash
uv run python -m unittest -v
```

## 📁 Project Structure

```
gaming-rag-research-agent/
├── main.py                          # Main application entry point
├── pyproject.toml                   # Project dependencies and metadata
├── README.md                        # This file
├── uv.lock                         # Dependency lock file
├── games/                          # Game dataset (JSON files)
│   ├── 001.json                    # Gran Turismo
│   ├── 002.json                    # Super Mario Bros
│   └── ...                         # Additional games (003-015)
├── lib/                            # Core library modules
│   ├── __init__.py
│   ├── agents.py                   # AI agent implementation
│   ├── documents.py                # Document processing
│   ├── evaluation.py               # Performance evaluation
│   ├── llm.py                      # Language model abstractions
│   ├── loaders.py                  # Data loading utilities
│   ├── memory.py                   # Memory management
│   ├── messages.py                 # Message handling
│   ├── parsers.py                  # Data parsing utilities
│   ├── rag.py                      # RAG implementation
│   ├── state_machine.py            # State management
│   ├── tooling.py                  # Tool implementations
│   └── vector_db.py                # Vector database interface
├── tests/                          # Test suite
│   └── test_memory_and_state_machine.py
├── chromadb/                       # ChromaDB persistence directory
└── *.ipynb                         # Jupyter notebooks for development
```

## 📄 License

This project is part of the Udacity Agentic AI Nanodegree program.


