from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Add the current directory to the path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from rag_system import RAGSystem
from pokemon_api import fetch_pokemon_data
from ragas_evaluator import RAGASEvaluator

app = Flask(__name__)
CORS(app)

# Initialize RAG system and RAGAS evaluator
openai_api_key = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
rag_system = RAGSystem(llm_api_key=openai_api_key)
ragas_evaluator = RAGASEvaluator(openai_api_key=openai_api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        personality_mode = data.get('personality', 'random')
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        # Classify the question type and extract Pokemon name if applicable
        question_type, pokemon_name = rag_system.classify_question(question)
        pokemon_data = None
        
        if question_type == "specific" and pokemon_name:
            # Specific Pokemon question
            pokemon_data = fetch_pokemon_data(pokemon_name)
            if 'error' in pokemon_data:
                # If the extracted Pokemon name doesn't exist, treat as general question
                pokemon_data = None
                question_type = "general"
                pokemon_name = None
        
        # Generate answer based on question type
        start_time = time.time()
        if question_type == "specific" and pokemon_data:
            # Pokemon-specific answer
            answer = rag_system.generate_answer(question, pokemon_data, personality_mode)
            context = ragas_evaluator.format_pokemon_context(pokemon_data)
        else:
            # General Pokemon knowledge answer
            answer = rag_system.generate_general_answer(question, personality_mode)
            context = "General Pokemon knowledge and information"
        
        response_time = round((time.time() - start_time) * 1000, 2)  # in milliseconds
        
        # Evaluate using RAGAS
        eval_scores = ragas_evaluator.evaluate_single_answer(question, answer, context)
        
        return jsonify({
            'answer': answer,
            'pokemon_name': pokemon_data.get('name', '').title() if pokemon_data else 'General Knowledge',
            'personality_mode': personality_mode,
            'response_time_ms': response_time,
            'eval_scores': eval_scores,
            'used_fallback': eval_scores.get('used_fallback', False),
            'pokemon_data': {
                'types': pokemon_data.get('types', []) if pokemon_data else ['general'],
                'weight': f"{pokemon_data.get('weight', 0) / 10} kg" if pokemon_data else 'N/A',
                'height': f"{pokemon_data.get('height', 0) / 10} m" if pokemon_data else 'N/A',
                'abilities': pokemon_data.get('abilities', []) if pokemon_data else ['knowledge']
            } if pokemon_data else {
                'types': ['general'],
                'weight': 'N/A',
                'height': 'N/A', 
                'abilities': ['knowledge']
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/variety_test', methods=['POST'])
def variety_test():
    """Test response variety by generating multiple answers to the same question"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        personality_mode = data.get('personality', 'random')
        num_responses = data.get('num_responses', 3)
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        # Classify the question type and extract Pokemon name if applicable
        question_type, pokemon_name = rag_system.classify_question(question)
        pokemon_data = None
        
        if question_type == "specific" and pokemon_name:
            pokemon_data = fetch_pokemon_data(pokemon_name)
            if 'error' in pokemon_data:
                # If the extracted Pokemon name doesn't exist, treat as general question
                pokemon_data = None
                question_type = "general"
                pokemon_name = None
        
        # Generate multiple responses using RAGAS evaluation
        start_time = time.time()
        
        if question_type == "specific" and pokemon_data:
            context = ragas_evaluator.format_pokemon_context(pokemon_data)
        else:
            context = "General Pokemon knowledge and information"
        
        answers = []
        
        for i in range(num_responses):
            if question_type == "specific" and pokemon_data:
                answer = rag_system.generate_answer(question, pokemon_data, personality_mode)
            else:
                answer = rag_system.generate_general_answer(question, personality_mode)
            answers.append(answer)
        
        # Evaluate with RAGAS
        ragas_results = ragas_evaluator.evaluate_variety_responses(question, answers, context)
        total_time = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'question': question,
            'pokemon_name': pokemon_data.get('name', '').title() if pokemon_data else 'General Knowledge',
            'personality_mode': personality_mode,
            'responses': ragas_results['responses'],
            'variety_score': ragas_results['variety_score'],
            'used_fallback': ragas_results.get('used_fallback', False),
            'total_time_ms': total_time,
            'pokemon_data': {
                'types': pokemon_data.get('types', []) if pokemon_data else ['general'],
                'weight': f"{pokemon_data.get('weight', 0) / 10} kg" if pokemon_data else 'N/A',
                'height': f"{pokemon_data.get('height', 0) / 10} m" if pokemon_data else 'N/A',
                'abilities': pokemon_data.get('abilities', []) if pokemon_data else ['knowledge']
            } if pokemon_data else {
                'types': ['general'],
                'weight': 'N/A',
                'height': 'N/A',
                'abilities': ['knowledge']
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/personalities')
def get_personalities():
    """Get available personality modes"""
    personalities = [
        {'value': 'random', 'label': 'Random', 'description': 'Varies between different personality styles'},
        {'value': 'enthusiastic', 'label': 'Enthusiastic', 'description': 'Excited and energetic responses'},
        {'value': 'scholarly', 'label': 'Scholarly', 'description': 'Academic and detailed responses'},
        {'value': 'casual', 'label': 'Casual', 'description': 'Relaxed and conversational responses'},
        {'value': 'factual', 'label': 'Factual', 'description': 'Clear and concise factual responses'}
    ]
    return jsonify(personalities)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
