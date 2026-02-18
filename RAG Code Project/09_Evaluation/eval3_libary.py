# 🧠 RAG EVALUATION TOOLS (OPEN-SOURCE + PAID)
# 🟢 OPEN-SOURCE TOOLS
# 1️⃣ RAGAS (Most popular – research + prod)


# pip install ragas datasets

# What it checks

# Faithfulness

# Answer Relevancy

# Context Precision

# Context Recall

# Minimal Code
from sys import monitoring
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset


data = {
    "question": ["Who is CEO of Tesla?"],
    "answer": ["Elon Musk is the CEO of Tesla."],
    "contexts": [["Elon Musk is the CEO of Tesla. Tesla was founded in 2003."]],
}


dataset = Dataset.from_dict(data)


result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)


print(result)


# Output
# faithfulness: 0.98
# answer_relevancy: 0.95
# How to decide
# Score	Decision
# > 0.8	Safe
# 0.5–0.8	Retry retrieval
# < 0.5	Block answer

# ⚠️ Uses LLM as judge → API cost

# 2️⃣ LlamaIndex Evaluation (If you use LlamaIndex)
# Install
# pip install llama-index
# Metrics

# Faithfulness

# Relevancy

# Correctness

# Code


from llama_index.core.evaluation import FaithfulnessEvaluator


evaluator = FaithfulnessEvaluator()


result = evaluator.evaluate_response(
    query="Who is CEO of Tesla?",
    response="Elon Musk is the CEO of Tesla.",
    contexts=["Elon Musk is the CEO of Tesla."]
)


print(result.passing, result.score)
# Output
# True 0.97
# Decision

# passing=False → hallucination → reject answer

# 3️⃣ DeepEval (CI/CD friendly)
# Install
# pip install deepeval
# Metrics

# FaithfulnessMetric

# AnswerRelevancyMetric

# ContextualPrecision

# Code
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase


test_case = LLMTestCase(
    input="Who is CEO of Tesla?",
    actual_output="Elon Musk is the CEO of Tesla.",
    retrieval_context=["Elon Musk is the CEO of Tesla."]
)


metric = FaithfulnessMetric()
metric.measure(test_case)


# print(metric.score)
# Output
# 0.96
# Why good

# Can fail tests in CI

# Lightweight

# Clean API

# 4️⃣ TruLens (Open-source + enterprise style)
# Install
# pip install trulens-eval
# Metrics

# Groundedness

# Relevance

# Coherence

# Code
from trulens_eval.feedback import Feedback
from trulens_eval.feedback.provider.openai import OpenAI


provider = OpenAI()
groundedness = Feedback(provider.groundedness_measure)


score = groundedness(
    answer="Elon Musk is the CEO of Tesla.",
    context="Elon Musk is the CEO of Tesla."
)


# print(score)
# Output
# 0.99
# Best for

# Dashboards

# Live monitoring

# 🔵 PAID / ENTERPRISE TOOLS
# 5️⃣ LangSmith (LangChain)
# Install
# pip install langsmith
# Setup
# export LANGCHAIN_API_KEY=xxxx
# What it does

# Trace RAG calls

# Eval faithfulness & relevance

# Compare runs

# Example Concept
from langsmith import traceable


@traceable
def rag_pipeline(query):
    ...
Decision

# Compare experiments

# Pick best retriever + prompt

# 6️⃣ UpTrain (Paid, enterprise monitoring)
# Install
# pip install uptrain
# Metrics

# Hallucination

# Groundedness

# Robustness

# Example

from uptrain import EvalLLM


evaluator = EvalLLM()
result = evaluator.evaluate(
    question="Who is CEO of Tesla?",
    answer="Elon Musk is CEO of Tesla",
    context="Elon Musk is CEO of Tesla"
)


print(result)
# Used for

# Online evaluation

# Alerts in production

# 7️⃣ Humanloop (Paid)
# Focus

# Human + LLM evaluation

# Feedback loops

# Used in:

# Startups

# Product teams

# 🧠 METRICS CHEAT SHEET (COMMON ACROSS TOOLS)
# Metric	Meaning
# Faithfulness	No hallucination
# Groundedness	Supported by context
# Answer Relevancy	Answers the question
# Context Precision	Low noise
# Context Recall	No missing info
# MRR / Hit@K	Retriever quality
# Stability	Consistent answers
# 🎯 HOW TO CHOOSE (IMPORTANT)
# Use this table 👇
# Your Case	Tool
# Research / blog	RAGAS
# LlamaIndex RAG	LlamaIndex Eval
# CI testing	DeepEval
# Live