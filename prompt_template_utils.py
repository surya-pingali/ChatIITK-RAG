"""
Prompt template for the chatbot. Simplified to a single generic instruction-style
template that works well with Phi-3 / Mistral / Llama-3 chat-tuned models via
the transformers text-generation pipeline (no llama.cpp specific token formatting needed,
since the tokenizer's chat template handles that).
"""

from langchain.prompts import PromptTemplate

system_prompt = """You are a helpful assistant (ChatIITK) that answers questions using ONLY the text given in the Context below.
Rules:
- Base your answer strictly on the Context. Do not use any outside or pretrained knowledge, even if you think you know the answer.
- If the Context does not contain enough information to answer, say "The provided context does not mention this."
- Do not invent names, places, or acronym expansions that are not explicitly present in the Context."""

instruction = """
Context:
{context}

Question: {question}

Answer (using only the Context above):"""

prompt_template = system_prompt + instruction

QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


def get_prompt_template():
    print(f"Here is the prompt used:\n{QA_PROMPT.template}")
    return QA_PROMPT