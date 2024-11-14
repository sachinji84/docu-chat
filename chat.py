import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "chromadb"  # This is path where your ChromaDB will be stored

# Initialize the OpenAI client with LangChain
llm_model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-4")

# In-memory storage for simplicity (consider using a database in production)
documents = {}

# In-memory storage for conversation history
conversation_history = {}

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the vector database globally at the start
vectordb = None

def initialize_vector_db():
    global vectordb
    if vectordb is None:
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
    return vectordb

# Initialize the vector store when the application starts
initialize_vector_db()

# Route to display upload page
@app.route('/')
def index():
    return render_template('upload.html')

# Handle PDF upload and process the document
@app.route('/upload', methods=['POST'])
def handle_upload():
    file = request.files.get('document')

    if file is None:
        return jsonify({"error": "No file part in the request"}), 400

    if file.filename.split('.')[-1].lower() != 'pdf':
        return jsonify({"error": "Only PDF files are allowed"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)

        vectordb.add_texts(
            texts=chunks,
            ids=[f"{file.filename}-{i}" for i in range(len(chunks))]
        )

        vectordb.persist()

        doc_id = len(documents) + 1
        documents[doc_id] = file.filename

        return redirect(url_for('chat', doc_id=doc_id))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to render chat page
@app.route('/chat/<int:doc_id>')
def chat(doc_id):
    if doc_id not in documents:
        return "Document not found", 404
    return render_template('chat.html', doc_id=doc_id)

# Handle chat interactions
@app.route('/interact/<int:doc_id>', methods=['POST'])
def interact_with_document(doc_id: int):
    if doc_id not in documents:
        return jsonify({"error": "Document not found"}), 404

    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    search_results = vectordb.similarity_search_with_score(user_query, k=5)
    relevant_content = "\n".join([result[0].page_content for result in search_results])

    history = conversation_history.get(doc_id, "")

    template = (
        "You are a helpful assistant.\n"
        "Conversation history:\n{history}\n"
        "Always answer from given context. If information is not available, say so.\n"
        "Relevant document content:\n{relevant_content}\n"
        "User query: {user_query}"
    )

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["history", "relevant_content", "user_query"]
    )

    chain = prompt_template | llm_model

    try:
        response = chain.invoke({
            "history": history,
            "relevant_content": relevant_content,
            "user_query": user_query
        })

        conversation_history[doc_id] = history + f"User: {user_query}\nAssistant: {response.content}\n"

        return jsonify({"response": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
