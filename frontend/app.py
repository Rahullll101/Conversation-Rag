import streamlit as st
import httpx
import json

# =========================
# CONFIG
# =========================
API_BASE_URL = "https://rag-backend-student-d7bqgvbxhuacaac2.southindia-01.azurewebsites.net"

st.set_page_config(
    page_title="Document Q&A RAG System",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# HELPERS
# =========================
def reset_session():
    st.session_state.session_id = None
    st.session_state.messages = []


# =========================
# TITLE
# =========================
st.title("Conversational Document Q&A RAG System")


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Upload Document")

    st.write(
        "Upload PDF or TXT files to build the knowledge base."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"]
    )

    if uploaded_file is not None:

        if st.button("Process Document"):

            reset_session()

            with st.spinner("Uploading and processing document..."):

                try:
                    # Detect MIME type
                    mime_type = (
                        "application/pdf"
                        if uploaded_file.name.lower().endswith(".pdf")
                        else "text/plain"
                    )

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            mime_type
                        )
                    }

                    # =========================
                    # STEP 1: Upload
                    # =========================
                    upload_response = httpx.post(
                        f"{API_BASE_URL}/upload",
                        files=files,
                        timeout=60.0
                    )

                    if upload_response.status_code != 200:
                        st.error(
                            f"Upload failed:\n{upload_response.text}"
                        )
                        st.stop()

                    upload_data = upload_response.json()

                    session_id = upload_data["session_id"]

                    # =========================
                    # STEP 2: Process
                    # =========================
                    process_response = httpx.post(
                        f"{API_BASE_URL}/process/{session_id}",
                        timeout=300.0
                    )

                    if process_response.status_code != 200:
                        st.error(
                            f"Processing failed:\n{process_response.text}"
                        )
                        st.stop()

                    process_data = process_response.json()

                    st.session_state.session_id = session_id

                    st.success(
                        f"Document processed successfully! "
                        f"{process_data['total_chunks']} chunks stored."
                    )

                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.divider()

    if st.session_state.session_id:
        st.success(
            f"Active Session: "
            f"{st.session_state.session_id[:8]}..."
        )

        if st.button("Upload New Document (Reset)"):

            try:
                httpx.delete(
                    f"{API_BASE_URL}/chat/memory/{st.session_state.session_id}",
                    timeout=10.0
                )
            except Exception:
                pass

            reset_session()
            st.rerun()

    else:
        st.warning("No active document session.")

    debug_mode = st.toggle("Debug Mode", value=False)

    eval_mode = st.toggle("Evaluation Mode", value=False)


# =========================
# EVALUATION MODE
# =========================
if eval_mode:

    st.subheader("Evaluation Dashboard")

    st.write(
        "Run the automated evaluation pipeline."
    )

    if st.button("Run Evaluation"):

        if not st.session_state.session_id:
            st.error(
                "Please upload and process a document first."
            )

        else:

            with st.spinner("Running evaluation pipeline..."):

                try:

                    response = httpx.post(
                        f"{API_BASE_URL}/evaluate/{st.session_state.session_id}",
                        timeout=600.0
                    )

                    if response.status_code == 200:

                        report = response.json()

                        st.success("Evaluation complete!")

                        st.metric(
                            "Aggregate Faithfulness",
                            f"{report.get('aggregate_faithfulness', 0):.2f}"
                        )

                        st.metric(
                            "Aggregate Relevancy",
                            f"{report.get('aggregate_relevancy', 0):.2f}"
                        )

                        st.dataframe([
                            {
                                "Question": entry["question"],
                                "Refused": entry["refused"],
                                "Faithfulness": entry["faithfulness"],
                                "Relevancy": entry["relevancy"]
                            }
                            for entry in report.get("entries", [])
                        ])

                    else:
                        st.error(response.text)

                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    st.stop()


# =========================
# CHAT HEADER
# =========================
col1, col2 = st.columns([0.8, 0.2])

with col1:
    st.subheader("Chat Interface")

with col2:

    if st.session_state.session_id:

        if st.button("🔄 New Chat", use_container_width=True):

            try:
                httpx.delete(
                    f"{API_BASE_URL}/chat/memory/{st.session_state.session_id}",
                    timeout=10.0
                )
            except Exception:
                pass

            st.session_state.messages = []
            st.rerun()


# =========================
# RENDER OLD MESSAGES
# =========================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        if "sources" in msg and msg["sources"]:

            with st.expander("View Sources"):

                for source in msg["sources"]:
                    st.json(source)

        if debug_mode and "debug_info" in msg:
            st.info(msg["debug_info"])


# =========================
# CHAT INPUT
# =========================
if query := st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.session_id
):

    # =========================
    # USER MESSAGE
    # =========================
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # =========================
    # ASSISTANT RESPONSE
    # =========================
    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:

                response_placeholder = st.empty()

                full_answer = ""

                with httpx.stream(
                    "POST",
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "query": query
                    },
                    timeout=300.0
                ) as response:

                    if response.status_code == 200:

                        metadata_json = ""

                        metadata_phase = False

                        for chunk in response.iter_text():

                            if "__METADATA__" in chunk:

                                split_parts = chunk.split("__METADATA__")

                                if split_parts[0]:
                                    full_answer += split_parts[0]

                                    response_placeholder.write(
                                        full_answer + "▌"
                                    )

                                metadata_phase = True

                                if len(split_parts) > 1:
                                    metadata_json += split_parts[1]

                                continue

                            if metadata_phase:
                                metadata_json += chunk

                            else:
                                full_answer += chunk

                                response_placeholder.write(
                                    full_answer + "▌"
                                )

                        response_placeholder.write(full_answer)

                        sources = []
                        debug_info = ""

                        if metadata_json.strip():

                            metadata = json.loads(metadata_json.strip())

                            sources = metadata.get("sources", [])

                            debug_info = (
                                f"Rewritten Query: "
                                f"{metadata.get('rewritten_query')}\n"
                                f"Memory Used: "
                                f"{metadata.get('memory_used')}"
                            )

                            if sources:

                                with st.expander(
                                    f"View {len(sources)} Sources"
                                ):

                                    for source in sources:

                                        source_copy = source.copy()

                                        if "session_id" in source_copy:
                                            del source_copy["session_id"]

                                        if "chunk_index" in source_copy:
                                            source_copy["Chunk Number"] = (
                                                source_copy.pop("chunk_index")
                                            )

                                        st.json(source_copy)

                            if debug_mode:
                                st.info(debug_info)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources,
                            "debug_info": debug_info
                        })

                    else:

                        response.read()

                        error_message = (
                            f"API Error:\n{response.text}"
                        )

                        st.error(error_message)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message
                        })

            except Exception as e:

                error_message = f"Request failed: {e}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })