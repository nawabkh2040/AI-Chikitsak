from setuptools import find_packages, setup

setup(
    name="Genrative AI Baesd Chatbot",
    version="0.0.0",
    author="Nawab Khan",
    author_email="nawabkh2040@gmail.com",
    packages=find_packages(),
    install_requires=["openai", "langchain", "langchain-pinecone", "langchain_community", "langchain_openai", "langchain_experimental", "sentence-transformers", "pinecone[grpc]", "pypdf", "flask", "python-dotenv"],
)
