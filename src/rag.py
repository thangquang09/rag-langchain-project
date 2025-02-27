from langchain import hub
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        self.str_parser = CustomStrOutputParser()
        self.no_data_message = "I apologize, but I couldn't find any relevant information in the provided documents to answer your question. Please try asking something related to the content of the documents."
        
    def get_chain(self, retriever):

        def check_context(input_dict):
            if not input_dict["context"].strip():
                return {
                    "context": "", 
                    "question": input_dict["question"], 
                    "no_data": True
                }
            return {**input_dict, "no_data": False}

        def generate_answer(input_dict):
            if input_dict.get("no_data", False):
                return self.no_data_message
            return self.llm.invoke(self.prompt.invoke(input_dict))

        rag_chain = (
            {
                "context": lambda x: self.format_docs(retriever.invoke(x)),
                "question": lambda x: x
            }
            | RunnablePassthrough.assign(no_data=lambda x: not x["context"].strip())
            | (lambda x: self.no_data_message if x["no_data"] else self.llm.invoke(self.prompt.invoke(x)))
            | self.str_parser
        )
        
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