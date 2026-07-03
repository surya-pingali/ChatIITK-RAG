import logging

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from constants import MAX_NEW_TOKENS, MODEL_ID, PERSIST_DIRECTORY, get_embeddings
from prompt_template_utils import get_prompt_template

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)

st.set_page_config(page_title="ChatIITK", page_icon="🎓")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner="Loading language model (first run downloads the model, please wait)...")
def load_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.15,
        return_full_text=False,
    )
    return pipe, tokenizer


def retrieve_context(db, question: str, k: int = 6) -> str:
    docs = db.similarity_search(question, k=k)
    seen, unique_texts = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_texts.append(d.page_content)
    return "\n\n".join(unique_texts)


def generate_answer(db, pipe, tokenizer, question: str) -> str:
    prompt = get_prompt_template()
    context = retrieve_context(db, question, k=6)
    filled_prompt_text = prompt.format(context=context, question=question)
    messages = [{"role": "user", "content": filled_prompt_text}]
    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(final_prompt)
    return result[0]["generated_text"].strip()


# ---- App UI ----
st.title("🎓 ChatIITK")
st.caption(f"Local RAG chatbot — running {MODEL_ID} on CPU, grounded in your SOURCE_DOCUMENTS.")

db = load_db()
pipe, tokenizer = load_llm_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask something about IIT Kanpur...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(db, pipe, tokenizer, user_question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})