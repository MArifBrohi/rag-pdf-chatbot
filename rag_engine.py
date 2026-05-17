import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Free local embeddings
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def load_and_index_pdf(pdf_paths: list):
    all_chunks = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = os.path.basename(pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
    vector_store.save_local("faiss_index")
    print("Vector store saved!")
    return vector_store


def load_vector_store():
    return FAISS.load_local(
        "faiss_index", EMBEDDINGS,
        allow_dangerous_deserialization=True
    )


def get_answer(question: str, vector_store, chat_history: list = []):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Groq LLM — free + fast!
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant that answers questions based on the provided PDF documents.
    
    Previous conversation:
    {chat_history}
    
    Context from PDF:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer ONLY based on the context provided
    - If answer is not in context, say "I couldn't find this in the uploaded documents"
    - Be concise but complete
    - Mention page numbers when possible
    
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join([
            f"[Page {doc.metadata.get('page', '?') + 1} | {doc.metadata.get('source_file', 'PDF')}]\n{doc.page_content}"
            for doc in docs
        ])

    def format_history(history):
        if not history:
            return "No previous conversation"
        return "\n".join([
            f"Human: {h['question']}\nAssistant: {h['answer']}"
            for h in history[-3:]
        ])

    source_docs = retriever.invoke(question)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": format_docs(source_docs),
        "question": question,
        "chat_history": format_history(chat_history)
    })

    return answer, source_docs