import os
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Model configuration
GPTModel = Literal["gpt-4o-mini"]
DEFAULT_MODEL = "gpt-4o-mini"


class QueryRequest(BaseModel):
    query: str
    model: GPTModel = DEFAULT_MODEL
    temperature: float = 0.7


# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
)

# Initialize Chroma
CHROMA_DB_PATH = "chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
vector_store = Chroma(
    client=client,
    collection_name="vehicles",
    embedding_function=embeddings,
)


def get_llm(model: GPTModel = DEFAULT_MODEL, temperature: float = 0.7):
    """Get LLM instance with specified model and temperature."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def process_csv(df: pd.DataFrame) -> List[Document]:
    """Process DataFrame into documents for the vector store."""
    documents = []

    for _, row in df.iterrows():
        # Enhanced content representation for better retrieval
        content = (
            f"This is a {row['make']} {row['model']} vehicle (ID: {row['vehicle']}). "
            f"It was manufactured in {row['fabrication_year'][:4]} and purchased on {row['date_bought']}. "
            f"This {row['make']} vehicle is part of the {row['model']} series. "
            f"This record can be used for counting {row['make']} vehicles and analyzing {row['make']} inventory."
        )

        metadata = {
            "vehicle": row["vehicle"],
            "date_bought": row["date_bought"],
            "make": row["make"],
            "fabrication_year": row["fabrication_year"],
            "model": row["model"]
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


@app.post("/query")
async def query_data(request: QueryRequest):
    try:
        llm = get_llm(request.model, request.temperature)

        template = """
        You are a vehicle database assistant. Answer the following question about vehicles based on the provided context.
        Focus on being precise and providing accurate counts and details.

        Important guidelines:
        - For counting queries, make sure to count ALL unique vehicles in the context
        - When asked about specific makes/models, include the total count if relevant
        - For list queries, organize the results clearly
        - If aggregating data, consider ALL provided examples
        - Always indicate if you're showing a partial result or if you have complete data
        - Numbers should be exact, not approximations

        Context: {context}
        Question: {question}

        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Enhanced retrieval strategy
        def smart_retriever(query):
            # Adjust k based on query complexity
            k = 10 if any(word in query.lower() for word in ['all', 'every', 'list']) else 5

            return vector_store.similarity_search(
                query,
                k=k,
            )

        retrieval_chain = (
            {"context": smart_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = retrieval_chain.invoke(request.query)

        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "model_used": request.model,
                "temperature": request.temperature
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))

        required_columns = ["vehicle", "date_bought", "make", "fabrication_year", "model"]
        if not all(col in df.columns for col in required_columns):
            return JSONResponse(
                status_code=400,
                content={"error": f"CSV must contain columns: {', '.join(required_columns)}"}
            )

        documents = process_csv(df)
        vector_store.add_documents(documents)

        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully processed {len(documents)} records"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
