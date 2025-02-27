from langchain import hub
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
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
        self.prompt = hub.pull("rlm/rag-prompt")
        self.embedding = HuggingFaceEmbeddings()
        self.str_parser = CustomStrOutputParser()
        self.no_data_message = "I apologize, but I couldn't find any relevant information in the provided documents to answer your question. Please try asking something related to the content of the documents."
        
    def get_chain(self, retriever):
        # Function to deduplicate documents
        def deduplicate_docs(docs, similarity_threshold=0.95):
            if not docs:
                return []
            embeddings = [self.embedding.embed_query(doc.page_content) for doc in docs]
            unique_docs = [docs[0]]
            for i in range(1, len(docs)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j] for j in range(len(unique_docs))])
                if max(sim[0]) < similarity_threshold:  # Nếu không quá giống
                    unique_docs.append(docs[i])
            return unique_docs
        
        # Create the RAG chain
        def rag_chain(query):
            # Retrieve and deduplicate documents
            docs = deduplicate_docs(retriever(query))
            
            # If no documents found, return the no_data_message
            if not docs:
                return self.no_data_message
            
            # Process with standard RAG pipeline
            context = self.format_docs(docs)
            prompt_result = self.prompt.invoke({"context": context, "question": query})
            llm_result = self.llm.invoke(prompt_result)
            return self.str_parser.invoke(llm_result)
        
        return rag_chain
        

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
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