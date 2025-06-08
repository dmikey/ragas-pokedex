# main.py

import json
import os
from dotenv import load_dotenv
from pokemon_api import fetch_pokemon_data
from rag_system import RAGSystem
from evaluator import evaluate_answers
from config import SAMPLE_QUESTIONS_FILE, EVALUATION_RESULTS_FILE

# Load environment variables
load_dotenv()

def load_sample_questions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    # Get OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables. Using placeholder.")
        api_key = "placeholder_key"
    
    # Initialize RAG system
    rag_system = RAGSystem(api_key)
    
    # Load sample questions from the JSON file
    questions = load_sample_questions(SAMPLE_QUESTIONS_FILE)
    
    # Store the generated answers and ground truth (basic facts from Pokemon API)
    generated_answers = {}
    ground_truth_answers = {}

    # Process each question
    for question_data in questions:
        pokemon_name = question_data['pokemon']
        query = question_data['question']
        
        # Retrieve data from the PokéAPI
        try:
            pokemon_data = fetch_pokemon_data(pokemon_name)
            
            # Generate an answer using the RAG system
            answer = rag_system.generate_answer(query, pokemon_data)
            
            # Store basic ground truth from Pokemon data for evaluation
            ground_truth = f"Pokemon {pokemon_data['name']}: {pokemon_data}"
            ground_truth_answers[query] = ground_truth
            generated_answers[query] = answer
            
            # Evaluate this specific answer immediately
            single_evaluation = evaluate_answers({query: ground_truth}, {query: answer})
            
            print(f"Q: {query}")
            print(f"A: {answer}")
            # print(f"Ground Truth: {ground_truth}")
            print(f"Evaluation Score: {single_evaluation['summary']['average_score']}")
            if query in single_evaluation['scores']:
                score_details = single_evaluation['scores'][query]
                print(f"  - Similarity Score: {score_details['similarity_score']}")
                print(f"  - Keyword Overlap: {score_details['keyword_overlap_score']}")
                print(f"  - Combined Score: {score_details['combined_score']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {pokemon_name}: {e}")

    # Evaluate all answers for overall statistics
    evaluation_results = evaluate_answers(ground_truth_answers, generated_answers)
    
    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS:")
    print(f"Average Score: {evaluation_results['summary']['average_score']}")
    print(f"Average Similarity: {evaluation_results['summary']['average_similarity']}")
    print(f"Average Keyword Overlap: {evaluation_results['summary']['average_keyword_overlap']}")
    print(f"Total Questions: {evaluation_results['summary']['total_questions']}")
    print(f"Valid Answers: {evaluation_results['summary']['valid_answers']}")
    print("=" * 60)

    # Save the evaluation results to a JSON file
    with open(EVALUATION_RESULTS_FILE, 'w') as file:
        json.dump(evaluation_results, file, indent=4)

    print("Processing completed. Evaluation results saved to:", EVALUATION_RESULTS_FILE)

if __name__ == "__main__":
    main()