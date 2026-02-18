"""
Docstring for Evaluation.post_processing_and_validation
🧩 MOST IMPORTANT POST-PROCESSING APPROACHES
1️⃣ Grounding Validation

Question: Is the answer supported by retrieved context?

2️⃣ Hallucination Detection

Question: Did the model add unsupported facts?

3️⃣ Answer Completeness

Question: Did it fully answer the user query?

4️⃣ Safety & Policy Check

Question: Is content safe to return?

5️⃣ Enhancements

Add citations

Highlight sources

Confidence score

Structured JSON output
"""
def normalize(text: str) -> str:
    return text.lower().strip()


# 1️⃣ GROUNDED IN CONTEXT CHECK (MOST IMPORTANT)

def validate_grounding(answer: str, contexts: list, llm) -> bool:
    context_text = "\n".join(contexts)

    prompt = f"""
Check whether the answer is fully supported by the context.

CONTEXT:
{context_text}

ANSWER:
{answer}

Respond only with:
YES or NO
"""
    result = llm(prompt).strip()
    return result == "YES"



# 2️⃣ HALLUCINATION DETECTION
def detect_hallucination(answer: str, contexts: list, llm) -> bool:
    prompt = f"""
Does the answer contain information not present in the context?

CONTEXT:
{contexts}

ANSWER:
{answer}

Respond only with:
HALLUCINATION or GROUNDED
"""
    result = llm(prompt).strip()
    return result == "HALLUCINATION"



# 3️⃣ ANSWER COMPLETENESS CHECK


def check_completeness(query: str, answer: str, llm) -> bool:
    prompt = f"""
Does the answer fully and clearly address the question?

QUESTION:
{query}

ANSWER:
{answer}

Respond only with:
COMPLETE or INCOMPLETE
"""
    result = llm(prompt).strip()
    return result == "COMPLETE"


# 4️⃣ SAFETY VALIDATION (GENERIC)

def safety_check(answer: str, llm) -> bool:
    prompt = f"""
Check if the answer contains unsafe, harmful, or disallowed content.

ANSWER:
{answer}

Respond only with:
SAFE or UNSAFE
"""
    result = llm(prompt).strip()
    return result == "SAFE"


# 5️⃣ ADD CITATIONS AUTOMATICALLY


def add_citations(answer: str, contexts: list, llm) -> str:
    prompt = f"""
Rewrite the answer and add citations like [CONTEXT X]
where each claim is supported.

CONTEXT:
{contexts}

ANSWER:
{answer}

CITED ANSWER:
"""
    return llm(prompt)



# 7️⃣ CONFIDENCE SCORING (ENTERPRISE FAVORITE)


def confidence_score(answer: str, contexts: list, llm) -> float:
    prompt = f"""
Give a confidence score between 0 and 1
based on how well the answer is supported by context.

CONTEXT:
{contexts}

ANSWER:
{answer}

Return only a number.
"""
    return float(llm(prompt))


def structured_response(answer, grounded, complete, safe, confidence, sources):
    return {
        "answer": answer,
        "grounded": grounded,
        "complete": complete,
        "safe": safe,
        "confidence": confidence,
        "sources": sources
    }

# FULL POST-PROCESSING PIPELINE (HOW COMPANIES DO IT)
def post_process_pipeline(query, answer, contexts, llm):
    grounded = validate_grounding(answer, contexts, llm)
    hallucinated = detect_hallucination(answer, contexts, llm)
    complete = check_completeness(query, answer, llm)
    safe = safety_check(answer, llm)

    if not grounded or hallucinated or not safe:
        return {
            "answer": "I don't have enough reliable information to answer this question.",
            "confidence": 0.0
        }

    answer = add_citations(answer, contexts, llm)
    confidence = confidence_score(answer, contexts, llm)
    sources = extract_sources(contexts)

    return structured_response(
        answer,
        grounded,
        complete,
        safe,
        confidence,
        sources
    )