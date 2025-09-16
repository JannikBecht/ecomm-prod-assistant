import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader

class DataIngestion:
    """
        Class to handle data transformation and ingestion into AstraDB vector store.
    """

    def __init__(self):
        print("initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_paht = self._get_csv_path()
        self.product_data = self._load_csv()
        self.conifg = load_config()

    def _load_env_variables(self):
        load_dotenv()

        required_vars = [
            "GOOGLE_API_KEY",
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE"
        ]

        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.astra_db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.astra_db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")


    def _get_csv_path(self):
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "data", "product_reviews.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        
        return csv_path

    def _load_csv(self):
        df = pd.read_csv(self.csv_paht)
        expected_columns = {'product_id','product_title', 'rating', 'total_reviews','price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def transform_data(self):
        product_list = []

        for _,row in self.product_data.iterrows():
            product_entry = {
                "product_id" : row["product_id"],
                "product_title": row["product_title"],
                "rating": row["rating"],
                "total_reviews": row["total_reviews"],
                "price": row["price"],
                "top_reviews": row["top_reviews"]
            }
        
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                "product_id": entry["product_id"],
                "product_title": entry["product_title"],
                "rating": entry["rating"],
                "total_reviews": entry["total_reviews"],
                "price": entry["price"]
            }

            doc = Document(page_content=entry["top_reviews"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents for ingestion.")
        return documents

    def store_in_vector_db(self):
        collection_name = self.conifg["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.astra_db_api_endpoint,
            token=self.astra_db_application_token,
            namespace=self.astra_db_keyspace
        )
        
        inserted_ids = vstore.add_documents(self.transform_data())
        print(f"Inserted {len(inserted_ids)} documents into AstraDB collection '{collection_name}'.")
        return vstore, inserted_ids
    
    def run_pipeline(self):
        documents = self.transform_data()
        vsstore, _ = self.store_in_vector_db(documents)

        query = "Can you tell me the low budget iphone?"
        results = vsstore.similarity_search(query)

        print(f"\nSample query results for '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()