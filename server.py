from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import faiss
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
faiss.omp_set_num_threads(32)

# Configure Google API
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "Google API Key not found. Please check your environment settings.")
genai.configure(api_key=google_api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)

# App configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pdf_content(file_path):
    text = ""
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", and do not provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_store_in_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("Vector store created and saved locally.")


def analyze_image(file_path):
    system_instruction = """
    You are tasked with analyzing an image. Focus on extracting key insights, trends, and visual patterns from the image. 
    Provide a detailed description and interpretation of the image content, ensuring that your explanation is concise yet thorough.
    """
    genai_keys = [
        "AIzaSyD4AvqSy5yE6FVIceijwFViKi76SObHsOY",
        "AIzaSyAPasufInx1YSA2N83orvuagkMe4ZnSOfE",
        'AIzaSyCu4O8kGxwU1BqGhlbiEnB-QQpEPzuEKfM',
        'AIzaSyCvb9F0bK_R4H14KDnWnbJeZSnIWvsDlAM'

    ]
    for idx, api_key in enumerate(genai_keys):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction,
            )
            myfile = genai.upload_file(file_path)
            response = model.generate_content(
                [myfile, "\n\n",
                    "Can you analyze this image and provide insights?"]
            )
            if response and hasattr(response, 'text'):
                return response.text
        except:
            logging.error("error rotating api keys")
            return None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"messages": ["No file part"]}), 400

        files = request.files.getlist('file')
        raw_text = ""

        with ThreadPoolExecutor() as executor:
            futures = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    if filename.lower().endswith(".pdf"):
                        futures.append(executor.submit(
                            get_pdf_content, file_path))
                    else:
                        futures.append(executor.submit(
                            analyze_image, file_path))

            for future in futures:
                result = future.result()
                if result:
                    if isinstance(result, str):
                        raw_text += result

        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_store_in_vector(text_chunks)
            return jsonify({"messages": ["Processing complete! You can now ask questions."]}), 200
        else:
            return jsonify({"messages": ["No text was extracted from the uploaded files. Please try again."]}), 400

    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question")
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        try:
            vector_store = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True)
            logging.info("Vector store loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            return jsonify({"messages": ["Error loading document index."]}), 500

        docs = vector_store.similarity_search(user_question)
        chain = get_conversation_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)

        output_text = response.get("output_text", "No response generated.")
        return jsonify({"messages": [output_text]}), 200

    return jsonify({"messages": ["No question provided."]}), 400


if __name__ == "__main__":
    app.secret_key = 'ANY_SECRET_KEY'
    app.run(debug=True)
