
"""
1️⃣ CONTEXT COVERAGE (Answer supported by retrieved chunks)

👉 Does the answer come from retrieved context?

Logic (simple)

If answer words appear in retrieved text → grounded
If not → hallucination risk

Code

"""
def context_coverage(answer, retrieved_chunks):
    context_text = " ".join(c["text"] for c in retrieved_chunks).lower()
    answer_tokens = set(answer.lower().split())


    covered = sum(1 for t in answer_tokens if t in context_text)
    return covered / max(len(answer_tokens), 1)


# Example Run
score = context_coverage(
    "Elon Musk is the CEO of Tesla",
    retrieved_chunks
)
print(score)

"""
Output
0.83
Meaning

✅ Most answer words exist in context
❌ If score < 0.5 → hallucination likely
"""


# 2️⃣ ANSWER FAITHFULNESS (LLM used context correctly)

"""
👉 Checks if LLM invented facts

Logic

Split answer into sentences → each must be supported by context

Code
"""

def faithfulness(answer, retrieved_chunks):
    context = " ".join(c["text"] for c in retrieved_chunks).lower()
    sentences = answer.lower().split(".")


    supported = 0
    for s in sentences:
        if s.strip() and any(word in context for word in s.split()):
            supported += 1


    return supported / max(len(sentences), 1)

"""
Output Meaning
Score	Meaning
> 0.7	Faithful
< 0.4	Hallucinating
3️⃣ ANSWER ↔ CONTEXT SIMILARITY (Retriever Quality)

👉 Measures if retrieved chunks are even related

Code (simple embedding-free)

"""

def answer_context_similarity(answer, retrieved_chunks):
    context_words = set(
        " ".join(c["text"] for c in retrieved_chunks).lower().split()
    )
    answer_words = set(answer.lower().split())


    return len(context_words & answer_words) / max(len(answer_words), 1)

"""

Low Score Means

❌ Retriever failed
❌ LLM guessed answer

4️⃣ CONTRADICTION CHECK (Silent Hallucinations)

👉 Does answer contradict retrieved text?

Code (basic)
"""
def contradiction_check(answer, retrieved_chunks):
    context = " ".join(c["text"] for c in retrieved_chunks).lower()


    negations = ["not", "no", "never"]
    for n in negations:
        if n in answer.lower() and n not in context:
            return 1  # possible contradiction


    return 0

# Output:

# 1 → risky answer

# 0 → safe



# 5️⃣ FINAL PRODUCTION SCORE (NO GOLD)
def rag_health_score(answer, retrieved_chunks):
    return {
        "coverage": context_coverage(answer, retrieved_chunks),
        "faithfulness": faithfulness(answer, retrieved_chunks),
        "similarity": answer_context_similarity(answer, retrieved_chunks),
        "contradiction": contradiction_check(answer, retrieved_chunks)
    }
# Run Everything Together
metrics = rag_health_score(final_answer, retrieved_chunks)


for k, v in metrics.items():
    print(k, ":", v)
# Example Output
# coverage : 0.83
# faithfulness : 1.0
# similarity : 0.78
# contradiction : 0
