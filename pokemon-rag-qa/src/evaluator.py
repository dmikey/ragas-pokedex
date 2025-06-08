# evaluator.py

import json
import re
from difflib import SequenceMatcher
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

def load_evaluation_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def similarity_score(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def keyword_overlap_score(generated, ground_truth):
    """Calculate keyword overlap score between generated answer and ground truth"""
    if not generated or not ground_truth:
        return 0.0
    
    # Extract words (simple tokenization)
    gen_words = set(re.findall(r'\w+', generated.lower()))
    truth_words = set(re.findall(r'\w+', ground_truth.lower()))
    
    if not truth_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = gen_words.intersection(truth_words)
    union = gen_words.union(truth_words)
    
    return len(intersection) / len(union) if union else 0.0

def evaluate_with_ragas(questions, answers, contexts, ground_truths):
    """Evaluate answers using RAGAS metrics"""
    
    # Prepare data for RAGAS evaluation
    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    
    # Convert to dataset
    dataset = Dataset.from_dict(data)
    
    # Define metrics to evaluate
    metrics = [
        answer_relevancy,
        faithfulness,
        answer_correctness,
        answer_similarity
    ]
    
    try:
        # Run RAGAS evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        return {
            'answer_relevancy': result['answer_relevancy'],
            'faithfulness': result['faithfulness'], 
            'answer_correctness': result['answer_correctness'],
            'answer_similarity': result['answer_similarity'],
            'overall_score': (
                result['answer_relevancy'] + 
                result['faithfulness'] + 
                result['answer_correctness'] + 
                result['answer_similarity']
            ) / 4
        }
    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        # Fallback to custom metrics
        return evaluate_answers_custom(questions, answers, ground_truths)

def evaluate_answers_custom(questions, answers, ground_truths):
    """Fallback custom evaluation when RAGAS fails"""
    total_similarity = 0
    total_keyword_overlap = 0
    valid_answers = len([a for a in answers if a and a.strip()])
    
    for i, (question, answer, ground_truth) in enumerate(zip(questions, answers, ground_truths)):
        if answer and answer.strip():
            sim_score = similarity_score(answer, ground_truth)
            keyword_score = keyword_overlap_score(answer, ground_truth)
            total_similarity += sim_score
            total_keyword_overlap += keyword_score
    
    avg_similarity = total_similarity / valid_answers if valid_answers > 0 else 0
    avg_keyword_overlap = total_keyword_overlap / valid_answers if valid_answers > 0 else 0
    
    return {
        'answer_relevancy': avg_similarity,
        'faithfulness': avg_keyword_overlap,
        'answer_correctness': (avg_similarity + avg_keyword_overlap) / 2,
        'answer_similarity': avg_similarity,
        'overall_score': (avg_similarity + avg_keyword_overlap) / 2
    }
def evaluate_answers(ground_truth_answers, generated_answers):
    """Legacy evaluate answers function - kept for backwards compatibility"""
    scores = {}
    total_similarity = 0
    total_keyword_overlap = 0
    valid_answers = 0
    
    for question in generated_answers.keys():
        generated_answer = generated_answers[question]
        ground_truth_answer = ground_truth_answers.get(question, "")
        
        if generated_answer and generated_answer.strip():
            # Calculate similarity score
            sim_score = similarity_score(generated_answer, ground_truth_answer)
            # Calculate keyword overlap score
            keyword_score = keyword_overlap_score(generated_answer, ground_truth_answer)
            
            # Combined score (weighted average)
            combined_score = (sim_score * 0.4 + keyword_score * 0.6)
            
            scores[question] = {
                "similarity_score": round(sim_score, 3),
                "keyword_overlap_score": round(keyword_score, 3),
                "combined_score": round(combined_score, 3),
                "answer_provided": True
            }
            
            total_similarity += sim_score
            total_keyword_overlap += keyword_score
            valid_answers += 1
        else:
            scores[question] = {
                "similarity_score": 0.0,
                "keyword_overlap_score": 0.0,
                "combined_score": 0.0,
                "answer_provided": False
            }
    
    # Calculate averages
    avg_similarity = total_similarity / valid_answers if valid_answers > 0 else 0
    avg_keyword_overlap = total_keyword_overlap / valid_answers if valid_answers > 0 else 0
    avg_combined = (avg_similarity * 0.4 + avg_keyword_overlap * 0.6)
    
    results = {
        "scores": scores,
        "summary": {
            "total_questions": len(generated_answers),
            "valid_answers": valid_answers,
            "average_similarity": round(avg_similarity, 3),
            "average_keyword_overlap": round(avg_keyword_overlap, 3),
            "average_score": round(avg_combined, 3)
        }
    }
    
    return results

def generate_evaluation_report(evaluation_results, output_path):
    with open(output_path, 'w') as file:
        json.dump(evaluation_results, file, indent=4)

if __name__ == "__main__":
    ground_truth_file = 'data/sample_questions.json'  # Adjust path as necessary
    generated_answers_file = 'data/evaluation_results.json'  # Adjust path as necessary
    output_report_file = 'data/evaluation_report.json'  # Adjust path as necessary

    ground_truth_answers = load_evaluation_data(ground_truth_file)
    generated_answers = load_evaluation_data(generated_answers_file)

    evaluation_results = evaluate_answers(generated_answers, ground_truth_answers)
    generate_evaluation_report(evaluation_results, output_report_file)

    print("Evaluation report generated successfully.")