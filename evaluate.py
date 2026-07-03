import time
import os

from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from constants import MAX_NEW_TOKENS, MODEL_ID, PERSIST_DIRECTORY, get_embeddings
from prompt_template_utils import get_prompt_template

torch.set_num_threads(os.cpu_count())

# Mock eval set: (question, list of keywords expected in retrieved context)
EVAL_SET = [
    ("What does E-Cell IIT Kanpur aim to do?", ["entrepreneurial", "start-ups", "incubators"]),
    ("What is E-Summit?", ["annual", "E-Summit", "participation"]),
    ("What is the mission of the Game Development Society?", ["popularize", "game development", "workshops"]),
    ("What autonomous underwater vehicles has Team AUV IIT Kanpur built?", ["Varun", "Anahita", "AUV"]),
    ("When was Team AUV IIT Kanpur started and how many members does it have?", ["2014", "40 members"]),
    ("What is IRASET and what does it work on?", ["Rocketry", "Solid-Propellant-Rocket-Motor", "student"]),
    ("What is the Academics and Career Council responsible for?", ["academics", "research", "Gymkhana"]),
    ("What wings does the Academics and Career Council have?", ["UG/PG Academics", "Research", "International Relations"]),
    ("What activities does the DesCon society organize?", ["popsicle bridge", "Spaghetti Bridge", "structural analysis"]),
    ("How does DesCon society teach design basics to students?", ["lectures", "workshops", "software"]),
]


def retrieve_context(db, question, k=6):
    docs = db.similarity_search(question, k=k)
    seen, unique_texts = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_texts.append(d.page_content)
    return "\n\n".join(unique_texts)


def main():
    print("Loading FAISS index...")
    embeddings = get_embeddings()
    db = FAISS.load_local(PERSIST_DIRECTORY, embeddings, allow_dangerous_deserialization=True)

    print("Loading LLM pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        repetition_penalty=1.15, return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt = get_prompt_template()

    retrieval_times, generation_times, hits = [], [], 0

    for question, keywords in EVAL_SET:
        t0 = time.perf_counter()
        context = retrieve_context(db, question, k=8)
        t1 = time.perf_counter()
        retrieval_times.append((t1 - t0) * 1000)

        hit = any(kw.lower() in context.lower() for kw in keywords)
        hits += int(hit)

        filled_prompt_text = prompt.format(context=context, question=question)
        messages = [{"role": "user", "content": filled_prompt_text}]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        t2 = time.perf_counter()
        pipe(final_prompt)
        t3 = time.perf_counter()
        generation_times.append((t3 - t2) * 1000)

        print(f"Q: {question} | Retrieval: {retrieval_times[-1]:.1f}ms | Generation: {generation_times[-1]:.1f}ms | Hit: {hit}")

    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_generation = sum(generation_times) / len(generation_times)
    hit_rate = (hits / len(EVAL_SET)) * 100

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Average Retrieval Latency: {avg_retrieval:.2f} ms")
    print(f"Average Generation Latency: {avg_generation:.2f} ms")
    print(f"Retrieval Hit-Rate: {hit_rate:.1f}%")
    print("===============================")


if __name__ == "__main__":
    main()