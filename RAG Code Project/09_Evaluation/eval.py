
"""


1️⃣ Retriever Function (returns chunks)

👉 Imagine this is your vector DB search
"""
def retrieve_context(query):
    """
    Simulated retriever (normally vector DB)
    """


    retrieved_chunks = [
        {"id": "c1", "text": "Elon Musk is the CEO of Tesla."},
        {"id": "c2", "text": "Tesla was founded in 2003."},
        {"id": "c3", "text": "Apple CEO is Tim Cook."}
    ]


    return retrieved_chunks


# 2️⃣ Generator Function (LLM uses retrieved chunks)

# 👉 This simulates LLM generation using context

def generate_answer(query, retrieved_chunks):
    """
    Simulated LLM generation
    """


    context_text = " ".join([c["text"] for c in retrieved_chunks])


    # Fake LLM logic
    if "CEO of Tesla" in query and "Elon Musk" in context_text:
        return "Elon Musk is the CEO of Tesla."


    return "I don't know."


# 3️⃣ RAG Pipeline (this is the real flow)


def rag_pipeline(query):
    retrieved_chunks = retrieve_context(query)
    answer = generate_answer(query, retrieved_chunks)


    return retrieved_chunks, answer


# 4️⃣ Run RAG (REAL EXECUTION)
query = "Who is the CEO of Tesla?"


retrieved_chunks, final_answer = rag_pipeline(query)


print("Retrieved Chunks:")
for c in retrieved_chunks:
    print("-", c["text"])


print("\nFinal Answer:")
print(final_answer)
# Output
# Retrieved Chunks:
# - Elon Musk is the CEO of Tesla.
# - Tesla was founded in 2003.
# - Apple CEO is Tim Cook.


# Final Answer:
# Elon Musk is the CEO of Tesla.
# 🔍 NOW EVALUATION STARTS (AFTER ANSWER)
# 5️⃣ Expected Correct Chunk (for evaluation)

gold_chunks = [
    {"id": "c1", "text": "Elon Musk is the CEO of Tesla."}
]


# 6️⃣ Evaluation Functions (REAL CHECKING)
# ✅ Context Recall
def context_recall(retrieved, gold):
    retrieved_ids = {c["id"] for c in retrieved}
    gold_ids = {c["id"] for c in gold}
    return len(retrieved_ids & gold_ids) / len(gold_ids)


# ✅ Context Precision
def context_precision(retrieved, gold):
    retrieved_ids = {c["id"] for c in retrieved}
    gold_ids = {c["id"] for c in gold}
    return len(retrieved_ids & gold_ids) / len(retrieved_ids)
# ✅ MRR


def mrr(retrieved, correct_id):
    for i, c in enumerate(retrieved):
        if c["id"] == correct_id:
            return 1 / (i + 1)
    return 0

# ✅ Hit Rate @ K
def hit_rate_at_k(retrieved, gold_ids, k=3):
    return int(any(c["id"] in gold_ids for c in retrieved[:k]))


# 7️⃣ Run Evaluation (CONNECTED TO PIPELINE)
recall = context_recall(retrieved_chunks, gold_chunks)
precision = context_precision(retrieved_chunks, gold_chunks)
mrr_score = mrr(retrieved_chunks, "c1")
hit3 = hit_rate_at_k(retrieved_chunks, {"c1"}, k=3)


print("\nEvaluation Results:")
print("Context Recall   :", recall)
print("Context Precision:", precision)
print("MRR              :", mrr_score)
print("Hit@3            :", hit3)
# Output
# Evaluation Results:
# Context Recall   : 1.0
# Context Precision: 0.33
# MRR              : 1.0
# Hit@3            : 1




