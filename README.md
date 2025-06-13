# Website Chatbot using Cohere and LangChain

A Streamlit-based chatbot that can intelligently interact with content from any public website using Retrieval-Augmented Generation (RAG) powered by Cohere's LLMs and embeddings.

Try it here: [https://olympicson-website-chat.streamlit.app/](https://olympicson-website-chat.streamlit.app/)

## ✨ Features
- Loads and splits web content for context-aware interaction
- Uses Cohere's latest LLMs and embeddings
- Implements RAG pipeline with context/history-aware query reformulation
- Clean chat UI using Streamlit

## 🚀 Tech Stack
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Cohere](https://cohere.com/)
- [ChromaDB](https://www.trychroma.com/)

## 📦 Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/akinsiraifedayo/chat-with-website-context-langchain/ website-chatbot
cd website-chatbot
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies with [uv](https://github.com/astral-sh/uv) (Recommended)**

`uv` is a super-fast Python package manager and resolver compatible with `pip` but much faster and more deterministic.

### Install `uv`

- **Linux/macOS**:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```
- **Windows (PowerShell):**
```powershell
iwr https://astral.sh/uv/install.ps1 -useb | iex
```

Then install dependencies:
```bash
uv pip install -r requirements.txt
```

4. **(Alternative) Install Dependencies with pip**

If you prefer traditional pip:
```bash
pip install -r requirements.txt
```

5. **Set Up Environment Variables**

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Update the `.env` file with your Cohere API key.

6. **Run the Application**

```bash
streamlit run app.py
```

## 📁 .env.example
```env
# Get your API key from https://dashboard.cohere.com/api-keys
COHERE_API_KEY=your_cohere_api_key_here
```

## 🔐 Notes
- You must have a valid Cohere API key to use this project.
- Make sure the website you input allows content scraping for educational or experimental purposes.

## 📄 License
MIT License

## 🤝 Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [Cohere](https://cohere.com/)
- [Streamlit](https://streamlit.io/)
