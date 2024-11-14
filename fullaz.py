import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from azure.ai.openai import ChatGPT, OpenAIClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
from azure.search.documents import SearchClient
from azure.search.documents.models import SearchQuery, QueryType
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchIndex,
    VectorSearch,
)

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Azure Credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
COSMOS_DB_CONNECTION_STRING = os.getenv("COSMOS_DB_CONNECTION_STRING")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")

# Initialize the Azure OpenAI client
openai_client = OpenAIClient(endpoint=AZURE_OPENAI_ENDPOINT, credential=AzureKeyCredential(AZURE_OPENAI_API_KEY))

# Initialize the Cosmos DB client
cosmos_client = CosmosClient(COSMOS_DB_CONNECTION_STRING)
cosmos_container = cosmos_client.get_database_client(COSMOS_DB_NAME).get_container_client(COSMOS_CONTAINER_NAME)

# Initialize the Azure Search client
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_API_KEY))

# Initialize the LangChain LLM with Azure OpenAI
llm_model = ChatOpenAI(client=openai_client, model_name="gpt-4")

# In-memory storage for simplicity (consider using a database in production)
documents = {}

# In-memory storage for conversation history
conversation_history = {}

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the vector database globally at the start (Cosmos DB for storing vectors)
vectordb = None

def initialize_vector_db():
    global vectordb
    if vectordb is None:
        embeddings = OpenAIEmbeddings(
            openai_api_key=AZURE_OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        vectordb = cosmos_container
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

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(content)

        # Embed document chunks and store them in Cosmos DB
        embeddings = OpenAIEmbeddings(openai_api_key=AZURE_OPENAI_API_KEY, model="text-embedding-ada-002")
        embedded_chunks = embeddings.embed_documents(chunks)

        # Store embeddings and metadata in Cosmos DB
        for i, chunk in enumerate(chunks):
            item = {
                'id': f"{file.filename}-{i}",
                'document': file.filename,
                'chunk': chunk,
                'embedding': embedded_chunks[i].tolist(),  # Convert embeddings to list for Cosmos DB
            }
            cosmos_container.create_item(body=item)

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

    try:
        # Perform semantic search using Azure AI Search
        search_results = search_client.search(
            search_text=user_query,
            query_type=QueryType.FULL,
            top=5  # Get top 5 results
        )

        relevant_content = "\n".join([result['chunk'] for result in search_results])
        if not relevant_content:
            return jsonify({"error": "No relevant content found in the document"}), 404

        # Get conversation history (initialize if None)
        history = conversation_history.get(doc_id, "")

        # Construct the prompt template
        prompt_template = (
            "You are a helpful assistant.\n"
            "Conversation history:\n{history}\n"
            "Always answer from given context. If information is not available, say so.\n"
            "Relevant document content:\n{relevant_content}\n"
            "User query: {user_query}"
        )

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
        return jsonify({"error": f"Error occurred while generating response: {str(e)}"}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
