import streamlit as st
import httpx

st.set_page_config(page_title="Document Q&A RAG System", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_session():
    st.session_state.session_id = None
    st.session_state.messages = []
    
st.title("Conversational Document Q&A RAG System")

with st.sidebar:
    st.header("Upload Document")
    st.write("Upload PDF or TXT files to add to the knowledge base.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            reset_session() # Clear old session on new document
            with st.spinner("Uploading and processing..."):
                try:
                    mime_type = "application/pdf" if uploaded_file.name.lower().endswith('.pdf') else "text/plain"
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
                    
                    # 1. Upload
                    upload_res = httpx.post("http://127.0.0.1:8000/upload", files=files, timeout=30.0)
                    if upload_res.status_code == 200:
                        session_id = upload_res.json()["session_id"]
                        
                        # 2. Process (Chunk & Embed)
                        process_res = httpx.post(f"http://127.0.0.1:8000/process/{session_id}", timeout=60.0)
                        if process_res.status_code == 200:
                            st.session_state.session_id = session_id
                            st.success(f"Processing complete! {process_res.json()['total_chunks']} chunks stored.")
                        else:
                            st.error(f"Processing failed: {process_res.text}")
                    else:
                        st.error(f"Upload failed: {upload_res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    
    st.divider()
    if st.session_state.session_id:
        st.success(f"Active Session: {st.session_state.session_id[:8]}...")
        if st.button("Upload New Document (Reset All)"):
            reset_session()
            st.rerun()
    else:
        st.warning("No active session.")
        
    debug_mode = st.toggle("Debug Mode", value=False)
    eval_mode = st.toggle("Evaluation Mode", value=False)

if eval_mode:
    st.subheader("Evaluation Dashboard")
    st.write("Run the automated Native LLM Judge evaluation pipeline against the sample queries.")
    if st.button("Run Evaluation"):
        if not st.session_state.session_id:
            st.error("Please upload and process a document first to run evaluation.")
        else:
            with st.spinner("Running evaluation pipeline..."):
                try:
                    res = httpx.post(f"http://127.0.0.1:8000/evaluate/{st.session_state.session_id}", timeout=600.0)
                    if res.status_code == 200:
                        report = res.json()
                        st.success("Evaluation complete! Reports saved to evaluation/results/")
                        
                        st.metric("Aggregate Faithfulness", f"{report.get('aggregate_faithfulness', 0):.2f}")
                        st.metric("Aggregate Relevancy", f"{report.get('aggregate_relevancy', 0):.2f}")
                        
                        st.dataframe([
                            {
                                "Question": e["question"],
                                "Refused": e["refused"],
                                "Faithfulness": e["faithfulness"],
                                "Relevancy": e["relevancy"]
                            } for e in report.get("entries", [])
                        ])
                    else:
                        st.error(f"Evaluation API failed: {res.text}")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
    st.stop() # Hide regular chat in eval mode

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.subheader("Chat Interface")
with col2:
    if st.session_state.session_id:
        if st.button("🔄 New Chat", use_container_width=True):
            # Clear backend memory
            try:
                httpx.delete(f"http://127.0.0.1:8000/chat/memory/{st.session_state.session_id}", timeout=5.0)
            except Exception:
                pass # Ignore connection errors on UI reset
            # Clear frontend memory
            st.session_state.messages = []
            st.rerun()

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                for s in msg["sources"]:
                    st.json(s)
        if debug_mode and "debug_info" in msg:
            st.info(f"Debug Info: {msg['debug_info']}")

# Chat input
if query := st.chat_input("Ask a question about your documents...", disabled=not st.session_state.session_id):
    # Display user query
    with st.chat_message("user"):
        st.write(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                answer_placeholder = st.empty()
                full_answer = ""
                
                with httpx.stream(
                    "POST",
                    "http://127.0.0.1:8000/chat",
                    json={"session_id": st.session_state.session_id, "query": query},
                    timeout=120.0
                ) as response:
                    if response.status_code == 200:
                        buffer = ""
                        metadata_json = ""
                        is_metadata_phase = False
                        
                        for chunk in response.iter_text():
                            buffer += chunk
                            
                            if "__METADATA__" in buffer:
                                parts = buffer.split("__METADATA__")
                                new_text = parts[0]
                                if new_text:
                                    full_answer += new_text
                                
                                is_metadata_phase = True
                                metadata_json += parts[1]
                                buffer = ""
                                continue
                                
                            if is_metadata_phase:
                                metadata_json += chunk
                            else:
                                full_answer += chunk
                                answer_placeholder.write(full_answer + "▌")
                                buffer = ""
                        
                        answer_placeholder.write(full_answer)
                        
                        import json
                        if metadata_json.strip():
                            data = json.loads(metadata_json.strip())
                            sources = data.get("sources", [])
                            
                            if sources:
                                with st.expander(f"View {len(sources)} Sources"):
                                    for s in sources:
                                        if "session_id" in s:
                                            del s["session_id"]
                                        if "chunk_index" in s:
                                            s["Chunk Number"] = s.pop("chunk_index")
                                        st.json(s)
                                        
                            debug_info = f"Rewritten Query: {data.get('rewritten_query')}\nMemory Used: {data.get('memory_used')}"
                            if debug_mode:
                                st.info(debug_info)
                                
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_answer, 
                                "sources": sources,
                                "debug_info": debug_info
                            })
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": full_answer})
                    else:
                        response.read()
                        err_msg = f"API Error: {response.text}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Request failed: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
