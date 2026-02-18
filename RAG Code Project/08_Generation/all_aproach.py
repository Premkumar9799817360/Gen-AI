

def format_context(contexts, max_chunks=5):
    selected = contexts[:max_chunks]
    return "\n\n".join(
        [f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(selected)]
    )


# 1️⃣ Stuffing Prompt Generation

def generate_stuffing(query, contexts, llm):
    context = format_context(contexts)

    prompt = f"""
You are an assistant.
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""
    return llm(prompt)


# 2️⃣ Map-Reduce Generation

def generate_map_reduce(query, contexts, llm):
    partial_answers = []


    for ctx in contexts:
        prompt = f"""
Answer the question using the context.


Context:
{ctx}


Question:
{query}
"""
        partial_answers.append(llm(prompt))


    reduce_prompt = f"""
Combine the following answers into one final answer.


Answers:
{partial_answers}


Final Answer:
"""
    return llm(reduce_prompt)

# 3️⃣ Refine Generation (VERY IMPORTANT STRATEGY)
def generate_refine(query, contexts, llm):
    answer = ""


    for i, ctx in enumerate(contexts):
        if i == 0:
            prompt = f"""
Answer the question using context.


Context:
{ctx}


Question:
{query}
"""
        else:
            prompt = f"""
Refine the existing answer using new context.


Existing Answer:
{answer}


New Context:
{ctx}


Refined Answer:
"""
        answer = llm(prompt)


    return answer



# 4️⃣ Context Compression Generation

def generate_compressed(query, contexts, llm):
    compression_prompt = f"""
Summarize the following context into key facts only.


Context:
{format_context(contexts)}


Summary:
"""
    compressed_context = llm(compression_prompt)


    final_prompt = f"""
Answer the question using the summary.


Summary:
{compressed_context}


Question:
{query}


Answer:
"""
    return llm(final_prompt)


# 5️⃣ Citation-Aware Generation (ENTERPRISE)

def generate_with_citations(query, contexts, llm):
    context = format_context(contexts)


    prompt = f"""
Answer the question using the context.
Cite sources as [CONTEXT X].


Context:
{context}


Question:
{query}


Answer with citations:
"""
    return llm(prompt)

# 6️⃣ Guardrail-Based Generation (ANTI-HALLUCINATION)

def generate_guarded(query, contexts, llm):
    context = format_context(contexts)


    prompt = f"""
RULES:
- Use ONLY the provided context
- If answer not found, say "Not found in context"
- Do NOT use prior knowledge


Context:
{context}


Question:
{query}


Answer:
"""
    return llm(prompt)


# 7️⃣ Controlled Reasoning (Hidden CoT Style)


def generate_reasoned(query, contexts, llm):
    context = format_context(contexts)


    prompt = f"""
Use the context to reason internally.
Return only the final answer.


Context:
{context}


Question:
{query}


Final Answer:
"""
    return llm(prompt)


# STRICT GUARDED PROMPT (BEST DEFAULT)
def generate_strict_guarded(query, contexts, llm):
    context = format_context(contexts)

    prompt = f"""
SYSTEM RULES:
- Use ONLY the provided context
- Do NOT use prior knowledge
- Do NOT guess or hallucinate
- If answer is missing, say "Not found in context"

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""
    return llm(prompt)




# CITATION-ENFORCED PROMPT (ENTERPRISE / TRUST)

def generate_with_citations(query, contexts, llm):
    context = format_context(contexts)

    prompt = f"""
SYSTEM RULES:
- Answer ONLY from context
- Cite each fact as [CONTEXT X]
- No unsupported statements

CONTEXT:
{context}

QUESTION:
{query}

ANSWER WITH CITATIONS:
"""
    return llm(prompt)



# REFINE PROMPT (LONG DOCUMENTS – MOST USED)

def generate_refine(query, contexts, llm):
    answer = ""

    for i, ctx in enumerate(contexts):
        if i == 0:
            prompt = f"""
Answer the question using the context.

CONTEXT:
{ctx}

QUESTION:
{query}

ANSWER:
"""
        else:
            prompt = f"""
Refine the existing answer using the new context.
Do not remove correct information.

EXISTING ANSWER:
{answer}

NEW CONTEXT:
{ctx}

REFINED ANSWER:
"""
        answer = llm(prompt)

    return answer




# CONTEXT COMPRESSION + ANSWER (TOKEN SAFE)s

def generate_compressed(query, contexts, llm):
    compression_prompt = f"""
Summarize the following context into factual bullet points.
Do NOT add information.

CONTEXT:
{format_context(contexts)}

SUMMARY:
"""
    summary = llm(compression_prompt)

    final_prompt = f"""
Use the summary below to answer the question.
If information is missing, say so.

SUMMARY:
{summary}

QUESTION:
{query}

ANSWER:
"""
    return llm(final_prompt)




# STRUCTURED OUTPUT PROMPT (API READY)

def generate_structured(query, contexts, llm):
    context = format_context(contexts)

    prompt = f"""
SYSTEM RULES:
- Use only the context
- Output valid JSON only
- No extra text

CONTEXT:
{context}

QUESTION:
{query}

OUTPUT FORMAT:
{{
  "answer": "...",
  "confidence": "high|medium|low"
}}
"""
    return llm(prompt)


# HALLUCINATION FAIL-SAFE PROMPT (CRITICAL SYSTEMS)

def generate_fail_safe(query, contexts, llm):
    context = format_context(contexts)

    prompt = f"""
CRITICAL RULE:
If the answer is NOT explicitly present in the context,
respond EXACTLY with:
"I don't have enough information in the provided context."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    return llm(prompt)



