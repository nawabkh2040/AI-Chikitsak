import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from flask import Flask, render_template, request, jsonify, session
from datetime import timedelta
import json
from utils.document_processor import process_documents
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize environment variables
PINECONE_API_KEY = os.getenv("PINECORN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set API keys
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Constants
INDEX_NAME = "ai-medi"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=1)

# Initialize document processor and retriever
print("Initializing document processor...")
retriever, embeddings = process_documents()

# Add image upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_file):
    """Process the uploaded image and return a base64 encoded string."""
    try:
        # Read the image file
        image = Image.open(image_file)
        
        # Resize if too large
        max_size = (800, 800)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64
        return base64.b64encode(img_byte_arr).decode('utf-8')
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def initialize_llm():
    """Initialize and return the OpenAI LLM."""
    try:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.7,  # Increased for more creative responses
            max_tokens=1000,  # Increased for more detailed answers
            model="gpt-3.5-turbo-instruct"  # Using a more capable model
        )
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise

def create_rag_chain(memory):
    """Create and return the RAG chain with memory."""
    try:
        # Initialize LLM
        llm = initialize_llm()

        # Create prompt template with memory
        system_prompt = (
            "You are a friendly and knowledgeable medical assistant. Your role is to provide clear, "
            "accurate, and easy-to-understand medical information. Follow these guidelines:\n\n"
            "1. Start with a warm, empathetic greeting\n"
            "2. Break down complex medical terms into simple language\n"
            "3. Use analogies and examples when helpful\n"
            "4. Structure your response with clear sections:\n"
            "   - Main Answer: 2-3 concise paragraphs\n"
            "   - Key Points: Bullet points of important information\n"
            "   - Additional Context: Any relevant background information\n"
            "5. End with a supportive closing statement\n"
            "6. Always cite your sources in a clear format\n\n"
            "IMPORTANT: If the question is outside your knowledge base or the provided context doesn't contain "
            "relevant information, respond with:\n"
            "\"I'm sorry, but I don't have enough information in my knowledge base to provide a reliable answer "
            "to your question. I recommend consulting with a healthcare professional for accurate medical advice.\"\n\n"
            "Remember to:\n"
            "- Be conversational but professional\n"
            "- Use simple language without medical jargon\n"
            "- Show empathy and understanding\n"
            "- Keep the response focused and relevant\n"
            "- Acknowledge when you don't have enough information\n\n"
            "Chat History:\n"
            "{chat_history}\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question: {input}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create chains with memory
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain

    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        raise

def get_answer(question: str, memory: ConversationBufferMemory) -> tuple:
    """Get answer for a given question using the RAG system with memory."""
    try:
        rag_chain = create_rag_chain(memory)
        
        # Prepare input with chat history
        input_data = {
            "input": question,
            "chat_history": memory.chat_memory.messages
        }
        
        response = rag_chain.invoke(input_data)
        
        # Extract answer and sources
        answer = response["answer"]
        sources = []
        
        # Check if the answer indicates lack of knowledge
        if "don't have enough information" in answer.lower() or "outside my knowledge base" in answer.lower():
            # Format the "I don't know" response
            formatted_answer = (
                "I'm sorry, but I don't have enough information in my knowledge base to provide a reliable answer "
                "to your question. I recommend consulting with a healthcare professional for accurate medical advice."
            )
            return formatted_answer, []
        
        # Extract sources from the answer
        if "Sources:" in answer:
            answer_parts = answer.split("Sources:")
            answer = answer_parts[0].strip()
            sources = [s.strip() for s in answer_parts[1].split("\n") if s.strip()]
        
        # Format the answer with HTML for better display
        formatted_answer = answer.replace("\n\n", "<br><br>")
        formatted_answer = formatted_answer.replace("\n", "<br>")
        
        return formatted_answer, sources
    except Exception as e:
        print(f"Error getting answer: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question.", []

def create_memory():
    """Create a new ConversationBufferMemory instance."""
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input",
        output_key="output"
    )
    return memory

def load_memory_from_session():
    """Load memory from session or create new if not exists."""
    if 'memory' not in session:
        memory = create_memory()
        save_memory_to_session(memory)
        return memory
    
    memory_data = json.loads(session['memory'])
    memory = create_memory()
    
    # Restore messages from session
    for msg in memory_data.get('chat_memory', {}).get('messages', []):
        if msg['type'] == 'human':
            memory.chat_memory.add_user_message(msg['content'])
        elif msg['type'] == 'ai':
            memory.chat_memory.add_ai_message(msg['content'])
    
    return memory

def save_memory_to_session(memory):
    """Save memory to session."""
    # Convert messages to a serializable format
    messages = []
    for msg in memory.chat_memory.messages:
        messages.append({
            'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
            'content': msg.content
        })
    
    session['memory'] = json.dumps({
        'chat_memory': {
            'messages': messages
        }
    })

@app.route('/')
def home():
    """Render the home page."""
    # Initialize session if not exists
    if 'memory' not in session:
        memory = create_memory()
        save_memory_to_session(memory)
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question submission and return answer."""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Load memory from session
        memory = load_memory_from_session()
        
        # Get answer and sources
        answer, sources = get_answer(question, memory)
        
        # Save context to memory
        memory.save_context({"input": question}, {"output": answer})
        
        # Save memory to session
        save_memory_to_session(memory)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'history': [{'type': 'human', 'content': msg.content} if isinstance(msg, HumanMessage) 
                       else {'type': 'ai', 'content': msg.content} 
                       for msg in memory.chat_memory.messages]
        })
    
    except Exception as e:
        print(f"Error in ask route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
