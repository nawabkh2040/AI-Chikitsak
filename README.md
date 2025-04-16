# AI-Chikitsak: AI-Powered Medical Assistant

AI-Chikitsak is an intelligent medical assistant that provides reliable medical information and guidance using advanced AI technologies. It combines the power of OpenAI's language models with a specialized medical knowledge base to deliver accurate and helpful responses to medical queries.

## Features

- 🤖 **Intelligent Medical Assistance**: Get accurate and reliable medical information
- 📚 **Knowledge Base Integration**: Access to a curated medical knowledge base
- 💬 **Conversational Interface**: Natural, easy-to-use chat interface
- 🔍 **Source Citations**: Transparent source references for all medical information
- 🚫 **Safety First**: Clear indication when information is outside the knowledge base
- 📱 **Responsive Design**: Works seamlessly on both desktop and mobile devices

## Setup Instructions

### Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.10
- OpenAI API key
- Pinecone API key

### Environment Setup

1. Create a new conda environment:
```bash
conda create -n openai-ml python=3.10 -y
```

2. Activate the environment:
```bash
conda activate openai-ml
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
PINECORN=your_pinecone_api_key
```

### Required Packages
- langchain==0.1.0
- langchain-community
- langchain-core
- langchain-openai
- pypdf==3.17.1
- python-dotenv==1.0.0
- openai==1.3.0
- pinecone
- langchain-pinecone
- flask
- sentence-transformers
- torch>=2.0.0
- transformers
- Pillow>=10.0.0

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Type your medical question in the chat interface and press Enter or click Send.

4. The AI assistant will provide a detailed response with:
   - Main answer
   - Key points
   - Additional context
   - Source citations

## Project Structure

```
AI-Chikitsak/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── templates/
│   └── index.html         # Frontend interface
├── utils/
│   └── document_processor.py  # Document processing utilities
└── research/
    └── stage_1.ipynb      # Research and development notebook
```

## Important Notes

- This is an AI assistant and not a replacement for professional medical advice
- Always consult with healthcare professionals for medical decisions
- The system will clearly indicate when it cannot provide reliable information
- Responses are based on the available knowledge base and may not cover all medical conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Acknowledgments

- OpenAI for providing the language model
- Pinecone for vector database services
- The medical community for their valuable knowledge and research


