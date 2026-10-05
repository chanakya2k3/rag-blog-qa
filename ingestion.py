import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

load_dotenv()

if __name__ == '__main__':
    #print("Ingesting...")
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "mediumblog1.txt"), encoding="utf-8")
    document  = loader.load()
    #print(document)
  
    #print("splitting...")
    
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    #print(texts)
          
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    #print("ingesting...")
    
    print(f"ingesting {len(texts)} chunks into '{os.environ['INDEX_NAME']}'...")
    PineconeVectorStore.from_documents(
        texts,
        embedding=embeddings,
        index_name=os.environ['INDEX_NAME']
    )
    print("done.")
                          
                                      