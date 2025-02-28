import streamlit as st
from constant import models, model_kwargs
from file_loader import Loader, get_num_cpu, get_file_paths
from vectordb import VectorDatabase
from llm import get_local_model, get_api_model
from rag import RAG
from utils import select_running_type, initial_data, check_data_exists
from pydantic import BaseModel, Field
import time

class InputQA(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class AnswerQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

# **Hàm giao diện chat**
def chat_interface(rag_chain):
    st.title("AI Assistant")
    st.write("Hỏi bất kỳ câu hỏi nào và nhận câu trả lời từ trợ lý AI.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..."):
                start_time = time.time()
                user_question = InputQA(question=prompt)
                result = rag_chain(query=user_question.question)
                time_taken = time.time() - start_time
                answer = AnswerQA(answer=result)
                st.markdown(answer.answer)
                st.write(f"Thời gian xử lý: {time_taken:.2f} giây")

        st.session_state.messages.append({"role": "assistant", "content": answer.answer})

# **Hàm tải tài liệu với caching**
@st.cache_data
def load_documents(_loader, _file_paths, _workers):
    documents = _loader.load(_file_paths, workers=_workers)
    return documents

# **Hàm khởi tạo vector database**
@st.cache_resource
def initialize_vectordb(_documents=None, load_new_vectordb=False, _cache_key=None):
    if _documents is not None and load_new_vectordb:
        vectordb = VectorDatabase(documents=_documents, load_new_vectordb=True)
    else:
        vectordb = VectorDatabase()
    return vectordb

# **Hàm chọn mô hình cục bộ**
def select_model_interface():
    # Hiển thị danh sách mô hình có sẵn
    st.sidebar.subheader("Mô hình cục bộ")
    
    # Option 1: Sử dụng mô hình có sẵn
    predefined_model = st.sidebar.selectbox(
        "Chọn mô hình có sẵn:",
        options=[f"{i}: {model}" for i, model in enumerate(models)],
        index=0
    )
    
    # Option 2: Nhập tên mô hình Hugging Face tùy chọn
    custom_model = st.sidebar.text_input("Hoặc nhập tên mô hình Hugging Face:", "")
    
    # Xác định mô hình được chọn
    if custom_model.strip():
        # Nếu người dùng đã nhập tên mô hình tùy chọn
        return custom_model.strip()
    else:
        # Lấy index từ predefined_model (format: "0: mistralai/Mistral-7B-Instruct-v0.1")
        selected_index = int(predefined_model.split(":")[0])
        return models[selected_index]

# **Hàm chính**
def main():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide"
    )
    # Check data and vector DB existence
    data_status = check_data_exists()
    
    # Handle different initialization scenarios
    if data_status != 0:
        message = {
            1: "Đang tải xuống dữ liệu PDF...",
            2: "Đang tạo Vector Database...",
            3: "Đang tải xuống dữ liệu PDF và tạo Vector Database..."
        }
        
        with st.spinner(message[data_status]):
            # Initialize data based on what's missing
            initial_data(data_status)
            
            success_message = {
                1: "Đã tải xuống dữ liệu PDF thành công!",
                2: "Đã tạo Vector Database thành công!",
                3: "Đã khởi tạo dữ liệu và Vector Database thành công!"
            }
            st.success(success_message[data_status])
    
    # Get file paths after initialization is complete (or if no initialization was needed)
    file_paths = get_file_paths()
    # Khởi tạo session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.rag_chain = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    loader = Loader()

    # Trong mã chính của bạn
    if st.sidebar.button("Tải lại tài liệu"):
        with st.spinner("Đang tải tài liệu..."):
            load_documents.clear()  # Xóa cache cũ nếu có
            documents = load_documents(loader, file_paths, get_num_cpu())
            cache_key = time.time()  # Tạo một khóa duy nhất bằng timestamp
            vectordb = initialize_vectordb(_documents=documents, load_new_vectordb=True, _cache_key=cache_key)
    else:
        vectordb = initialize_vectordb()

    retriever = vectordb.get_retriever()

    # **Chọn loại mô hình**
    running_type = st.sidebar.radio("Chọn loại mô hình:", ("Local Model", "API Model"), index=0)
    
    # Tạo các phần chọn mô hình dựa trên loại
    if running_type == "Local Model":
        model_name = select_model_interface()
    elif running_type == "API Model":
        model_name = "API Model"  # Placeholder for API model
    
    # Tạo nút tải mô hình
    if st.sidebar.button("Tải và khởi động mô hình"):
        with st.spinner(f"Đang tải mô hình {model_name}..."):
            # Tải mô hình phù hợp
            if running_type == "Local Model":
                llm = get_local_model(model_name=model_name, **model_kwargs)
            elif running_type == "API Model":
                llm = get_api_model(**model_kwargs)
                
            # Tạo RAG Chain
            st.session_state.rag_chain = RAG(llm=llm).get_chain(retriever=retriever)
            st.session_state.model_loaded = True
            st.sidebar.success(f"Đã tải xong mô hình {model_name}")
    
        # Thêm nút xóa lịch sử chat vào sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quản lý cuộc hội thoại")
    if st.sidebar.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        st.sidebar.success("Đã xóa lịch sử chat")
        st.rerun()  # Thay thế st.experimental_rerun() bằng st.rerun()

    # Hiển thị thông báo nếu chưa tải mô hình
    if not st.session_state.model_loaded:
        st.info("Vui lòng chọn mô hình và nhấn 'Tải và khởi động mô hình' để bắt đầu sử dụng.")
    else:
        # Load giao diện chat
        chat_interface(st.session_state.rag_chain)

if __name__ == "__main__":
    main()