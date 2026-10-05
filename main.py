import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
)

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

#print(llm.invoke("Say hello in one word.").content)

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
  
#create a retrieval chain using LCEL(LangChain Expression Language)

def create_retrieval_chain_with_lcel():
  
  retrieval_chain = (
    RunnablePassthrough.assign(
      context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
  )
  return retrieval_chain

if __name__ == "__main__":
    # Create the retrieval chain using LCEL
    retrieval_chain = create_retrieval_chain_with_lcel()

    # Example usage of the retrieval chain
    question = "can you tell important insights from the context?"
    result = retrieval_chain.invoke({"question": question})
    print(result)