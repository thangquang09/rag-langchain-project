# from langchain import hub

# prompt = hub.pull("rlm/rag-prompt")
# print("\n"*5)
# print(prompt.invoke({
#     "context": "aaa",
#     "question": "bbb"
# }))


from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
        "Based on the following context:\n{context}\n,"
        "Provide a concise answer to the question '{question}' without repeating the same details multiple times.")
print(prompt_template.invoke({"context": "cats", "question": "question"}))