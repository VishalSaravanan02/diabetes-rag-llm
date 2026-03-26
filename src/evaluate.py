import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from src.retriever import retrieve
from src.generator import generate_answer
from config import TOP_K


# Test Dataset
# 10 diabetes questions with ground truth answers
TEST_QUESTIONS = [
    {
        "question": "What is diabetes mellitus?",
        "ground_truth": "Diabetes mellitus is a chronic metabolic disorder characterised by high blood sugar levels resulting from defects in insulin secretion, insulin action, or both."
    },
    {
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "ground_truth": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. Type 2 diabetes is characterised by insulin resistance and impaired insulin secretion, and is often associated with obesity and lifestyle factors."
    },
    {
        "question": "What are the main risk factors for Type 2 diabetes?",
        "ground_truth": "Main risk factors for Type 2 diabetes include obesity, physical inactivity, family history, age, high blood pressure, high cholesterol, and ethnicity."
    },
    {
        "question": "What is insulin resistance?",
        "ground_truth": "Insulin resistance is a condition where cells in the body do not respond effectively to insulin, leading to elevated blood glucose levels and forcing the pancreas to produce more insulin to compensate."
    },
    {
        "question": "What are the complications of diabetes?",
        "ground_truth": "Diabetes complications include kidney damage (nephropathy), nerve damage (neuropathy), eye damage (retinopathy), cardiovascular disease, foot ulcers, and increased risk of infections."
    },
    {
        "question": "What is HbA1c and why is it important in diabetes?",
        "ground_truth": "HbA1c is a measure of average blood glucose levels over the past 2 to 3 months. It is used to diagnose diabetes and monitor how well blood sugar is being controlled in people with diabetes."
    },
    {
        "question": "What is diabetic kidney disease?",
        "ground_truth": "Diabetic kidney disease, also known as diabetic nephropathy, is a complication of diabetes that damages the kidneys filtering system, potentially leading to kidney failure if left untreated."
    },
    {
        "question": "How does obesity relate to Type 2 diabetes?",
        "ground_truth": "Obesity, particularly excess abdominal fat, contributes to insulin resistance which is a key factor in the development of Type 2 diabetes. Weight loss can significantly improve insulin sensitivity and blood sugar control."
    },
    {
        "question": "What is gestational diabetes?",
        "ground_truth": "Gestational diabetes is a type of diabetes that develops during pregnancy, usually in the second or third trimester. It occurs when the body cannot produce enough insulin to meet the increased demands of pregnancy."
    },
    {
        "question": "What lifestyle changes can help manage Type 2 diabetes?",
        "ground_truth": "Lifestyle changes that help manage Type 2 diabetes include regular physical exercise, a healthy balanced diet, weight loss, quitting smoking, limiting alcohol, and monitoring blood glucose levels regularly."
    }
]

# Model cache check
HF_CACHE  = Path.home() / ".cache" / "huggingface" / "hub"
BGE_MODEL = "models--BAAI--bge-large-en-v1.5"


def _check_models():
    """
    Verify that required local models are available before running evaluation.
    Raises a clear error if a model needs to be downloaded first.
    """
    if not (HF_CACHE / BGE_MODEL).exists():
        raise EnvironmentError(
            "BAAI/bge-large-en-v1.5 model not found in local cache.\n"
            "Please download it first with:\n\n"
            "    huggingface-cli download BAAI/bge-large-en-v1.5\n"
        )
    print("✓ BAAI/bge-large-en-v1.5 found in local cache.")


def build_eval_dataset():
    """
    Run each test question through the full RAG pipeline to collect
    questions, generated answers, retrieved contexts and ground truths.
    """
    print("\nRunning test questions through RAG pipeline...")
    print(f"Total questions: {len(TEST_QUESTIONS)}\n")

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, item in enumerate(TEST_QUESTIONS, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        # Retrieve relevant chunks
        results = retrieve(question, top_k=TOP_K)

        # Extract chunk texts for context
        retrieved_contexts = [r["chunk"] for r in results] if results else ["No context found."]

        # Generate answer from context
        context = "\n\n".join(retrieved_contexts)
        answer  = generate_answer(question, context)

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

        print(f"    ✓ Answer generated, {len(retrieved_contexts)} chunks retrieved\n")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })


def run_evaluation():
    """
    Evaluate the RAG pipeline using RAGAS metrics with local models:
    - LLM:        llama3 via Ollama
    - Embeddings: BAAI/bge-large-en-v1.5 (must be downloaded first via
                  huggingface-cli download BAAI/bge-large-en-v1.5)
    """

    # Check models are available
    _check_models()

    # Set up local LLM and embeddings for RAGAS
    print("\nLoading evaluation models...")

    # Use Ollama llama3 as the evaluation LLM
    eval_llm = LangchainLLMWrapper(ChatOllama(model="llama3"))

    # Uses the locally cached BAAI/bge-large-en-v1.5 model
    # Download it first with: huggingface-cli download BAAI/bge-large-en-v1.5
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    )

    # Configure metrics
    metrics = [
        Faithfulness(llm=eval_llm),
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        ContextPrecision(llm=eval_llm),
        ContextRecall(llm=eval_llm)
    ]

    # Build evaluation dataset
    print("\nBuilding evaluation dataset...")
    dataset = build_eval_dataset()

    # Run evaluation
    print("Running RAGAS evaluation...")
    print("(This may take a few minutes)\n")

    results = evaluate(dataset=dataset, metrics=metrics)

    # Display results
    print("\n" + "="*50)
    print("       RAGAS EVALUATION RESULTS")
    print("="*50)

    scores = results.to_pandas()

    print(f"\n📊 Aggregate Scores:")
    print(f"   Faithfulness:      {scores['faithfulness'].mean():.4f}")
    print(f"   Answer Relevancy:  {scores['answer_relevancy'].mean():.4f}")
    print(f"   Context Precision: {scores['context_precision'].mean():.4f}")
    print(f"   Context Recall:    {scores['context_recall'].mean():.4f}")

    print(f"\n📋 Per Question Scores:")
    print("-"*50)
    for i, row in scores.iterrows():
        print(f"\nQ{i+1}: {TEST_QUESTIONS[i]['question']}")
        print(f"   Faithfulness:      {row['faithfulness']:.4f}")
        print(f"   Answer Relevancy:  {row['answer_relevancy']:.4f}")
        print(f"   Context Precision: {row['context_precision']:.4f}")
        print(f"   Context Recall:    {row['context_recall']:.4f}")

    # Save results to disk
    output_path = "data/evaluation_results.json"
    results_dict = scores.to_dict(orient="records")

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Full results saved to {output_path}")
    print("="*50)

    return results


if __name__ == "__main__":
    run_evaluation()