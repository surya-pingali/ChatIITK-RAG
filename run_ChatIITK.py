import logging

import torch
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from constants import MAX_NEW_TOKENS, MODEL_ID, PERSIST_DIRECTORY, get_embeddings
from prompt_template_utils import get_prompt_template

# ---- Hardcoded test query (no manual terminal input required) ----
TEST_QUERY = "What is BCS?"

def load_llm_pipeline():
    logging.info(f"Loading tokenizer and model for {MODEL_ID} (this may take a while on first run)...")
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
        do_sample=False,          # deterministic, less prone to wandering off-context
        temperature=None,
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

def main():
    logging.info(f"Loading FAISS index from {PERSIST_DIRECTORY}")
    embeddings = get_embeddings()
    db = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)

    pipe, tokenizer = load_llm_pipeline()
    prompt = get_prompt_template()

    context = retrieve_context(db, TEST_QUERY, k=4)

    # --- Debug: confirm retrieval is actually pulling relevant chunks ---
    print("\n" + "-" * 60)
    print("RETRIEVED CONTEXT (debug):")
    print(context[:1500])
    print("-" * 60 + "\n")

    filled_prompt_text = prompt.format(context=context, question=TEST_QUERY)

    # Use TinyLlama's chat template instead of a raw string prompt
    messages = [{"role": "user", "content": filled_prompt_text}]
    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    logging.info("Generating answer...")
    result = pipe(final_prompt)
    answer = result[0]["generated_text"].strip()

    print("\n" + "=" * 60)
    print(f"QUESTION: {TEST_QUERY}")
    print("-" * 60)
    print(f"ANSWER: {answer}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()