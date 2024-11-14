import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from azure.ai.openai import ChatGPT, OpenAIClient
from azure.core.credentials import AzureKeyCredential

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Use your Azure OpenAI key and endpoint
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

CHROMA_DB_PATH = "chromadb"  # Path where your ChromaDB will be stored

# Initialize the Azure OpenAI client
openai_client = OpenAIClient(endpoint=AZURE_OPENAI_ENDPOINT, credential=AzureKeyCredential(AZURE_OPENAI_API_KEY))

# Initialize the LangChain LLM with Azure OpenAI
llm_model = ChatOpenAI(client=openai_client, model_name="gpt-4")

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
            openai_api_key=AZURE_OPENAI_API_KEY,
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

    # Perform similarity search
    try:
        search_results = vectordb.similarity_search_with_score(user_query, k=5)
        if search_results is None:
            print("Search results are None.")
            return jsonify({"error": "No relevant content found for the query"}), 404
        if len(search_results) == 0:
            print("Search results are empty.")
            return jsonify({"error": "No relevant content found for the query"}), 404

        print(f"Found {len(search_results)} results.")

        # Extract relevant content from search results
        relevant_content = "\n".join([result[0].page_content for result in search_results])
        print(f"Relevant content: {relevant_content[:200]}...")  # Print the first 200 characters

        if not relevant_content:
            print("No relevant content found.")
            return jsonify({"error": "No relevant content found in the document"}), 404

        # Get conversation history (initialize if None)
        history = conversation_history.get(doc_id, "")
        print(f"Conversation history: {history}")

        # Construct the prompt template
        prompt_template = (
            "You are a helpful assistant.\n"
            "Conversation history:\n{history}\n"
            "Always answer from given context. If information is not available, say so.\n"
            "Relevant document content:\n{relevant_content}\n"
            "User query: {user_query}"
        )
        print(f"Prompt template: {prompt_template}")

        # Execute the prompt using Azure OpenAI
        response = openai_client.completion(
            model="gpt-4",
            prompt=prompt_template.format(
                history=history,
                relevant_content=relevant_content,
                user_query=user_query
            ),
            max_tokens=150
        )

        assistant_reply = response.result.choices[0].text.strip()

        # Update conversation history with the new interaction
        conversation_history[doc_id] = history + f"User: {user_query}\nAssistant: {assistant_reply}\n"

        return jsonify({"response": assistant_reply})

    except Exception as e:
        print(f"Error: {e}")  # Print the error to logs for better debugging
        return jsonify({"error": f"Error occurred while generating response: {str(e)}"}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
