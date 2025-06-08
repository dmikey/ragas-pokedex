# ragas_evaluator.py

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness,
    answer_similarity
)
import os
from openai import OpenAI

class RAGASEvaluator:
    def __init__(self, openai_api_key=None):
        """Initialize RAGAS evaluator with OpenAI API key"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            # Set environment variable for RAGAS to use
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
    
    def evaluate_single_answer(self, question, answer, context, ground_truth=None):
        """Evaluate a single Q&A pair using RAGAS metrics"""
        
        # Prepare data for RAGAS
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [[context]],  # RAGAS expects list of contexts
            'ground_truth': [ground_truth] if ground_truth else [answer]  # Use answer as ground truth if none provided
        }
        
        try:
            # Convert to dataset
            dataset = Dataset.from_dict(data)
            
            # Define metrics to evaluate
            metrics = [
                answer_relevancy,
                faithfulness,
                answer_correctness,
                answer_similarity
            ]
            
            # Run RAGAS evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics
            )
            
            # Extract scores from EvaluationResult object
            print(f"RAGAS result type: {type(result)}")
            
            # Get the scores from the first (and only) row
            if hasattr(result, 'scores') and len(result.scores) > 0:
                scores_dict = result.scores[0]  # Get first row of scores
                print(f"Successfully extracted scores: {list(scores_dict.keys())}")
            else:
                raise ValueError("No scores found in RAGAS result")
            
            # Extract scores - handle numpy types
            def extract_score(value):
                if hasattr(value, 'item'):  # numpy types
                    return float(value.item())
                elif isinstance(value, (list, tuple)):
                    return float(value[0]) if len(value) > 0 else 0.0
                return float(value)
            
            faithfulness_score = extract_score(scores_dict['faithfulness'])
            correctness_score = extract_score(scores_dict['answer_correctness'])
            relevancy_score = extract_score(scores_dict['answer_relevancy'])
            # Handle different similarity key names
            if 'answer_similarity' in scores_dict:
                similarity_score = extract_score(scores_dict['answer_similarity'])
            elif 'semantic_similarity' in scores_dict:
                similarity_score = extract_score(scores_dict['semantic_similarity'])
            else:
                similarity_score = 0.7  # Default fallback
            
            scores = {
                'factual_accuracy': faithfulness_score,
                'response_quality': correctness_score,
                'engagement': relevancy_score,
                'similarity': similarity_score,
                'overall': (faithfulness_score + correctness_score + relevancy_score + similarity_score) / 4,
                'used_fallback': False
            }
            
            return scores
            
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            # Fallback to simple scoring
            fallback_scores = self._fallback_scoring(question, answer, context, ground_truth)
            fallback_scores['used_fallback'] = True
            return fallback_scores
    
    def evaluate_variety_responses(self, question, answers, context, ground_truth=None):
        """Evaluate multiple responses for variety testing"""
        results = []
        any_fallback_used = False
        
        for i, answer in enumerate(answers):
            scores = self.evaluate_single_answer(question, answer, context, ground_truth)
            if scores.get('used_fallback', False):
                any_fallback_used = True
            results.append({
                'response_index': i + 1,
                'answer': answer,
                'eval_scores': scores
            })
        
        # Calculate variety score based on response diversity
        variety_score = self._calculate_variety_score(answers)
        
        return {
            'responses': results,
            'variety_score': variety_score,
            'used_fallback': any_fallback_used
        }
    
    def _calculate_variety_score(self, responses):
        """Calculate variety score based on lexical diversity"""
        if len(responses) < 2:
            return 0.0
        
        all_words = []
        unique_words = set()
        
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)
            unique_words.update(words)
        
        if not all_words:
            return 0.0
            
        variety_score = len(unique_words) / len(all_words)
        return round(variety_score, 3)
    
    def _fallback_scoring(self, question, answer, context, ground_truth):
        """Enhanced fallback scoring with distinct metrics when RAGAS fails"""
        
        # Tokenize for analysis
        answer_words = answer.lower().split()
        question_words = question.lower().split()
        context_words = context.lower().split() if context else []
        
        # === RESPONSE QUALITY: Structure, completeness, and clarity ===
        word_count = len(answer_words)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        pokemon_terms = ['pokemon', 'type', 'ability', 'move', 'stat', 'evolution', 'species', 'generation']
        pokemon_term_count = sum(1 for term in pokemon_terms if term in answer.lower())
        
        # Quality components
        length_quality = min(word_count / 50.0, 1.0) * 0.8 if word_count >= 10 else word_count / 10.0
        structure_quality = min(sentence_count / 3.0, 1.0)  # Good responses have multiple sentences
        pokemon_relevance = min(pokemon_term_count / 3.0, 1.0)  # Uses Pokemon-specific terminology
        
        response_quality = (length_quality + structure_quality + pokemon_relevance) / 3.0
        
        # === ENGAGEMENT: Personality, enthusiasm, and interactivity ===
        exclamation_count = answer.count('!')
        question_count = answer.count('?')
        engaging_words = ['amazing', 'incredible', 'fantastic', 'awesome', 'cool', 'wow', 'great', 'wonderful', 'exciting', 'love']
        enthusiasm_words = sum(1 for word in engaging_words if word in answer.lower())
        personal_pronouns = sum(1 for word in ['you', 'your', 'we', 'let\'s'] if word in answer.lower())
        
        # Engagement components
        enthusiasm_score = min(exclamation_count / 2.0, 1.0) * 0.4
        personality_score = min(enthusiasm_words / 2.0, 1.0) * 0.4
        interactivity_score = min((question_count + personal_pronouns) / 2.0, 1.0) * 0.2
        
        engagement = enthusiasm_score + personality_score + interactivity_score
        
        # === FACTUAL ACCURACY: Pokemon knowledge and context consistency ===
        factual_score = 0.5  # default neutral
        if context_words and answer_words:
            # Check overlap between context facts and answer (meaningful words only)
            stop_words = {'is', 'are', 'the', 'a', 'an', 'what', 'how', 'does', 'do'}
            context_set = set(context_words) - stop_words
            answer_set = set(answer_words) - stop_words
            overlap = len(context_set.intersection(answer_set))
            factual_score = min(overlap / max(len(context_set), 1) * 1.5, 1.0)
        
        # === SIMILARITY: Lexical and semantic overlap ===
        similarity_score = 0.7  # default
        if ground_truth:
            # Word overlap similarity with ground truth
            ground_words = set(ground_truth.lower().split())
            answer_word_set = set(answer_words)
            if ground_words:
                similarity_score = len(ground_words.intersection(answer_word_set)) / len(ground_words)
        elif context_words:
            # Similarity to context when no ground truth
            context_set = set(context_words)
            answer_set = set(answer_words)
            similarity_score = min(len(context_set.intersection(answer_set)) / max(len(answer_set), 1) * 2, 1.0)
        
        # Calculate overall score
        overall_score = (factual_score + response_quality + engagement + similarity_score) / 4
        
        return {
            'factual_accuracy': round(factual_score, 3),
            'response_quality': round(response_quality, 3),
            'engagement': round(engagement, 3),
            'similarity': round(similarity_score, 3),
            'overall': round(overall_score, 3),
            'used_fallback': False  # This will be overridden to True when called from exception handler
        }
    
    def format_pokemon_context(self, pokemon_data):
        """Format Pokemon data as context for RAGAS evaluation"""
        context = f"""
Pokemon: {pokemon_data.get('name', 'Unknown')}
Types: {', '.join(pokemon_data.get('types', []))}
Weight: {pokemon_data.get('weight', 0) / 10} kg ({pokemon_data.get('weight', 0)} hectograms)
Height: {pokemon_data.get('height', 0) / 10} m ({pokemon_data.get('height', 0)} decimeters)
Abilities: {', '.join(pokemon_data.get('abilities', []))}
        """.strip()
        return context
