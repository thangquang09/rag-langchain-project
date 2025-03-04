from constant import prompt

# from langchain import hub
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import re

class CustomStrOutputParser(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_ans(text)
    
    def extract_ans(self, text: str, pattern: str = "Answer:\s*(.*)") -> str:
        match = re.search(pattern=pattern, string=text, flags=re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            return text

class RAG:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = prompt
        self.embedding = HuggingFaceEmbeddings()
        self.str_parser = CustomStrOutputParser()
        self.store = {}

    def get_store(self):
        return self.store
    # Function to deduplicate documents
    def deduplicate_docs(self, docs, similarity_threshold=0.95):
        if not docs:
            return []
        embeddings = [self.embedding.embed_query(doc.page_content) for doc in docs]
        unique_docs = [docs[0]]
        for i in range(1, len(docs)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j] for j in range(len(unique_docs))])
            if max(sim[0]) < similarity_threshold:  # Nếu không quá giống
                unique_docs.append(docs[i])
        return unique_docs

    def get_chain(self, retriever):
        
        # Create the RAG chain
        def rag_chain(query):
            # Retrieve and deduplicate documents
            docs = self.deduplicate_docs(retriever(query))
            
            # Process with standard RAG pipeline
            context = self.format_docs(docs)
            prompt_result = self.prompt.invoke({"context": context, "question": query})
            llm_result = self.llm.invoke(prompt_result)
            return self.str_parser.invoke(llm_result)
        
        return rag_chain
    
    def get_chain_with_history(self, retriever):
        """Create a RAG chain with simple manual chat history support"""
        # Store for chat histories
        # store = {}
        
        def rag_chain(query, config=None):
            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", session_id)
            
            # Initialize or get history
            if session_id not in self.store:
                self.store[session_id] = []
            
            history = self.store[session_id]
            
            # Retrieve and deduplicate documents
            docs = self.deduplicate_docs(retriever(query))
            
            # Process with standard RAG pipeline
            context = self.format_docs(docs)
            prompt_result = self.prompt.invoke({
                "context": context, 
                "question": query, 
                "chat_history": history
            })
            llm_result = self.llm.invoke(prompt_result)
            result = self.str_parser.invoke(llm_result)
            
            # Update history (simplified)
            self.store[session_id].append({"type": "human", "content": query})
            self.store[session_id].append({"type": "ai", "content": result})
            
            return result
        
        return rag_chain
    
    def format_docs(self, docs):
        """Format documents and remove duplicates based on content"""
        seen = set()
        unique_docs = []
        
        for doc in docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)
        
        return "\n\n".join(doc.page_content for doc in unique_docs)
    


if __name__ == "__main__":
    pass