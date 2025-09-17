import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config
from dotenv import load_dotenv

class Retriever:

    def __init__(self):
        pass

    def _load_env_variables(self):
        """
        Load environment variables from .env file in local mode.
        In production, it assumes environment variables are set externally.
        """
        pass

    def load_retriever(self):
        """
        Load and return a vector store retriever.
        """
        pass

    def call_retriever(self, user_query: str) -> List[Document]:
        """
        Call the retriever to fetch relevant documents.
        """
        pass

if __name__ == "__main__":
    retriever = Retriever()
    user_query = "Can you suggest good budget laptops?"
    results = retriever.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata{doc.metadata}\n")