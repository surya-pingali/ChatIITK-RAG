import logging
import time
import os

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from constants import MAX_NEW_TOKENS, MODEL_ID, PERSIST_DIRECTORY, get_embeddings
from prompt_template_utils import get_prompt_template

torch.set_num_threads(os.cpu_count())
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ChatIITK API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def load_resources():
    logging.info("Loading FAISS index...")
    embeddings = get_embeddings()
    _state["db"] = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)

    logging.info("Loading LLM pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.15,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    _state["pipe"] = pipe
    _state["tokenizer"] = tokenizer
    logging.info("Startup complete.")


def retrieve_context(db, question: str, k: int = 6):
    docs = db.similarity_search(question, k=k)
    seen, unique_texts = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_texts.append(d.page_content)
    return "\n\n".join(unique_texts)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieval_latency_ms: float
    generation_latency_ms: float


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    db = _state["db"]
    pipe = _state["pipe"]
    tokenizer = _state["tokenizer"]
    prompt = get_prompt_template()

    t0 = time.perf_counter()
    context = retrieve_context(db, req.question, k=8)
    t1 = time.perf_counter()

    filled_prompt_text = prompt.format(context=context, question=req.question)
    messages = [{"role": "user", "content": filled_prompt_text}]
    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t2 = time.perf_counter()
    result = pipe(final_prompt)
    t3 = time.perf_counter()

    answer = result[0]["generated_text"].strip()

    return ChatResponse(
        answer=answer,
        retrieval_latency_ms=(t1 - t0) * 1000,
        generation_latency_ms=(t3 - t2) * 1000,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}