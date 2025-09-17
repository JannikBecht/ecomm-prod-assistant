import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv

class Retriever:

    def __init__(self):
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        """
        Load environment variables from .env file in local mode.
        In production, it assumes environment variables are set externally.
        """
        load_dotenv()
         
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def load_retriever(self):
        """
        Load and return a vector store retriever.
        """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            
            self.vstore =AstraDBVectorStore(
                embedding= self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
                )
            
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever = self.vstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            return retriever
        
        return self.retriever

    def call_retriever(self, user_query: str) -> List[Document]:
        """
        Call the retriever to fetch relevant documents.
        """
        retriever = self.load_retriever()
        results = retriever.invoke(user_query)
        return results

if __name__ == "__main__":
    retriever = Retriever()
    user_query = "Can you suggest good budget laptops?"
    results = retriever.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata{doc.metadata}\n")