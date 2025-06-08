# rag_system.py

from openai import OpenAI
from pokemon_api import fetch_pokemon_data

class RAGSystem:
    def __init__(self, llm_api_key):
        self.client = OpenAI(api_key=llm_api_key)

    def generate_answer(self, question, pokemon_data, personality_mode="random"):
        """Generate an answer based on the question and Pokemon data
        
        Args:
            question: The user's question
            pokemon_data: Data about the Pokemon
            personality_mode: "random", "enthusiastic", "scholarly", "casual", or "factual"
        """
        
        # If we have a valid OpenAI API key, use it
        if self.client.api_key and self.client.api_key != "your_openai_api_key_here":
            # Define personality-specific prompts
            personality_prompts = {
                "enthusiastic": "You are an incredibly excited Pokémon trainer who LOVES talking about Pokémon! Use lots of exclamation points and energetic language!",
                "scholarly": "You are a distinguished Pokémon researcher who speaks in an academic, detailed manner with scientific precision.",
                "casual": "You are a chill Pokémon fan who talks like you're chatting with a friend. Keep it relaxed and conversational.",
                "factual": "You are a Pokémon database that provides clear, concise, factual information without embellishment.",
                "random": "You are a passionate Pokémon expert with a fun, engaging personality. Vary your response style randomly between enthusiastic, scholarly, and casual approaches."
            }
            
            # Use the specified personality or default to random
            system_prompt = personality_prompts.get(personality_mode, personality_prompts["random"])
            system_prompt += """ Answer questions about Pokémon using the provided data accurately, but bring personality to your responses. Make each answer unique and engaging while staying factually correct."""
            
            user_prompt = f"""Question: {question}
            
Pokémon Data: {pokemon_data}

IMPORTANT UNIT CONVERSIONS:
- Height is given in decimeters in the raw data. To convert to standard units:
  * Decimeters to meters: divide by 10 (e.g., 5 decimeters = 0.5 meters)
  * Decimeters to centimeters: multiply by 10 (e.g., 5 decimeters = 50 cm)
  * Decimeters to inches: multiply by ~4 (e.g., 5 decimeters = ~20 inches)
- Weight is given in hectograms. To convert to kilograms: divide by 10

Please answer this question using the provided data with CORRECT unit conversions. Make your response engaging and unique while being factually accurate!"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=200,  # Increased for more detailed responses
                    temperature=0.9,  # Increased for more creativity and variation
                    top_p=0.95,      # Add nucleus sampling for more diverse outputs
                    frequency_penalty=0.3,  # Reduce repetitive phrases
                    presence_penalty=0.3    # Encourage new topics/words
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error generating answer: {e}"
        
        # Otherwise, use rule-based answers based on Pokemon data
        question_lower = question.lower()
        
        if "type" in question_lower:
            types = pokemon_data.get('types', [])
            if types:
                return f"{pokemon_data['name'].title()} is a {'/'.join(types)} type Pokémon."
            
        elif "weight" in question_lower or "weigh" in question_lower:
            weight = pokemon_data.get('weight', 0)
            weight_kg = weight / 10  # Convert from hectograms to kg
            return f"{pokemon_data['name'].title()} weighs {weight_kg} kg."
            
        elif "height" in question_lower or "tall" in question_lower:
            height = pokemon_data.get('height', 0)
            height_m = height / 10  # Convert from decimeters to meters
            return f"{pokemon_data['name'].title()} is {height_m} meters tall."
            
        elif "ability" in question_lower or "abilities" in question_lower:
            abilities = pokemon_data.get('abilities', [])
            if abilities:
                ability_list = ', '.join(abilities)
                return f"{pokemon_data['name'].title()} has the following abilities: {ability_list}."
                
        elif "evolve" in question_lower:
            # This is more complex and would require additional API calls
            return f"Evolution information for {pokemon_data['name'].title()} would require additional API queries to the evolution chain endpoint."
            
        elif "move" in question_lower:
            return f"Move information for {pokemon_data['name'].title()} would require additional API queries to get the complete moveset."
            
        elif "habitat" in question_lower:
            return f"Habitat information for {pokemon_data['name'].title()} would require additional API queries to the species endpoint."
            
        elif "color" in question_lower:
            return f"Color information for {pokemon_data['name'].title()} would require additional API queries to the species endpoint."
            
        elif "experience" in question_lower:
            return f"Base experience information for {pokemon_data['name'].title()} would require additional data from the main Pokémon API response."
        
        # Default response
        return f"Here's what I know about {pokemon_data['name'].title()}: Types: {'/'.join(pokemon_data.get('types', []))}, Weight: {pokemon_data.get('weight', 0)/10}kg, Height: {pokemon_data.get('height', 0)/10}m, Abilities: {', '.join(pokemon_data.get('abilities', []))}"

    def process_question(self, question, personality_mode="random"):
        pokemon_name = self.extract_pokemon_name(question)
        pokemon_data = fetch_pokemon_data(pokemon_name)
        generated_answer = self.generate_answer(question, pokemon_data, personality_mode)
        return generated_answer

    def classify_question(self, question):
        """Classify if a question is about a specific Pokémon or general knowledge using OpenAI"""
        if not question:
            return "general", None
        
        # If we have a valid OpenAI API key, use it for intelligent classification
        if self.client.api_key and self.client.api_key != "your_openai_api_key_here":
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a Pokémon question classifier. Your job is to determine if a question is about:
1. A specific Pokémon (extract the Pokémon name)
2. General Pokémon knowledge

Return your response in this exact JSON format:
{
  "type": "specific" or "general",
  "pokemon_name": "pokemon_name_if_specific" or null
}

Examples:
- "What type is Charizard?" → {"type": "specific", "pokemon_name": "charizard"}
- "How many Pokémon types are there?" → {"type": "general", "pokemon_name": null}
- "Tell me about Pikachu's abilities" → {"type": "specific", "pokemon_name": "pikachu"}
- "What are legendary Pokémon?" → {"type": "general", "pokemon_name": null}"""
                        },
                        {
                            "role": "user",
                            "content": f"Classify this question: {question}"
                        }
                    ],
                    max_tokens=50,
                    temperature=0.1  # Low temperature for consistent classification
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Try to parse the JSON response
                try:
                    import json
                    result = json.loads(result_text)
                    question_type = result.get("type", "general")
                    pokemon_name = result.get("pokemon_name")
                    
                    # Clean up pokemon name if provided
                    if pokemon_name:
                        pokemon_name = pokemon_name.lower().strip()
                    
                    return question_type, pokemon_name
                except json.JSONDecodeError:
                    # Fallback to the old method if JSON parsing fails
                    return self._fallback_classification(question)
                    
            except Exception as e:
                # Fallback to the old method if OpenAI fails
                return self._fallback_classification(question)
        else:
            # Fallback to the old method if no API key
            return self._fallback_classification(question)

    def _fallback_classification(self, question):
        """Fallback classification method using pattern matching"""
        if not question:
            return "general", None
        
        question_lower = question.lower()
        
        # Common Pokemon names (subset for fallback)
        known_pokemon = [
            'pikachu', 'charizard', 'bulbasaur', 'squirtle', 'charmander', 'ivysaur',
            'venusaur', 'wartortle', 'blastoise', 'eevee', 'vaporeon', 'jolteon',
            'flareon', 'snorlax', 'mewtwo', 'mew', 'articuno', 'zapdos', 'moltres',
            'dragonite', 'gyarados', 'lapras', 'ditto', 'gengar', 'alakazam', 'machamp'
        ]
        
        # Check for known Pokemon names
        for pokemon in known_pokemon:
            if pokemon in question_lower:
                return "specific", pokemon
        
        # Check for general question patterns
        general_patterns = [
            "how many", "what are", "which", "all", "list", "different",
            "types of", "regions", "generations", "legendary", "mythical",
            "starter", "gym", "elite four", "champion", "shiny"
        ]
        
        for pattern in general_patterns:
            if pattern in question_lower:
                return "general", None
        
        # Try to extract potential Pokemon name from remaining words
        words = question_lower.split()
        stop_words = {'what', 'how', 'much', 'does', 'is', 'the', 'of', 'can', 'learn', 
                     'type', 'weight', 'height', 'ability', 'abilities', 'moves', 'move',
                     'habitat', 'color', 'experience', 'base', 'evolve', 'evolves', 'from',
                     'weigh', 'tall', 'which', 'pokemon', 'pokémon', '?', 'a', 'an'}
        
        filtered_words = []
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word not in stop_words and len(clean_word) > 2:
                filtered_words.append(clean_word)
        
        # If we found potential Pokemon names, assume it's specific
        if filtered_words:
            return "specific", filtered_words[0]
        
        # Default to general if we can't determine
        return "general", None

    def extract_pokemon_name(self, question):
        """Extract Pokemon name from question - kept for backward compatibility"""
        question_type, pokemon_name = self.classify_question(question)
        return pokemon_name if question_type == "specific" else None

    def evaluate_answers(self, questions, ground_truth_answers, personality_mode="random", test_variety=False, ragas_evaluator=None):
        """Evaluate answers using RAGAS, optionally testing response variety
        
        Args:
            questions: List of questions to evaluate
            ground_truth_answers: Dict of expected answers
            personality_mode: Personality to use for responses
            test_variety: If True, generates multiple responses per question to test variety
            ragas_evaluator: RAGASEvaluator instance for evaluation
        """
        evaluation_results = {}
        
        for question in questions:
            # Extract Pokemon and get data for context
            pokemon_name = self.extract_pokemon_name(question)
            if pokemon_name:
                from pokemon_api import fetch_pokemon_data
                pokemon_data = fetch_pokemon_data(pokemon_name)
                
                if 'error' not in pokemon_data and ragas_evaluator:
                    context = ragas_evaluator.format_pokemon_context(pokemon_data)
                    ground_truth = ground_truth_answers.get(question, None)
                    
                    if test_variety:
                        # Generate multiple responses to test variety
                        responses = []
                        for i in range(3):  # Generate 3 different responses
                            response = self.generate_answer(question, pokemon_data, personality_mode)
                            responses.append(response)
                        
                        ragas_results = ragas_evaluator.evaluate_variety_responses(question, responses, context, ground_truth)
                        evaluation_results[question] = {
                            "responses": ragas_results['responses'],
                            "variety_score": ragas_results['variety_score'],
                            "ground_truth": ground_truth,
                            "pokemon_data": pokemon_data,
                            "context": context
                        }
                    else:
                        generated_answer = self.generate_answer(question, pokemon_data, personality_mode)
                        eval_scores = ragas_evaluator.evaluate_single_answer(question, generated_answer, context, ground_truth)
                        evaluation_results[question] = {
                            "generated_answer": generated_answer,
                            "eval_scores": eval_scores,
                            "ground_truth": ground_truth,
                            "pokemon_data": pokemon_data,
                            "context": context
                        }
                else:
                    # Fallback when Pokemon data not found
                    evaluation_results[question] = {
                        "error": f"Could not find Pokemon data for question: {question}",
                        "ground_truth": ground_truth_answers.get(question, "N/A")
                    }
            else:
                evaluation_results[question] = {
                    "error": f"Could not extract Pokemon name from question: {question}",
                    "ground_truth": ground_truth_answers.get(question, "N/A")
                }
                
        return evaluation_results

    def check_correctness(self, generated_answer, ground_truth):
        return generated_answer.lower() == ground_truth.lower() if ground_truth != "N/A" else False

    def calculate_variety_score(self, responses):
        """Calculate a variety score based on how different the responses are"""
        if len(responses) < 2:
            return 0.0
        
        # Simple variety metric: ratio of unique words across all responses
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

    def generate_general_answer(self, question, personality_mode="random"):
        """Generate an answer for general Pokémon knowledge questions
        
        Args:
            question: The user's question
            personality_mode: "random", "enthusiastic", "scholarly", "casual", or "factual"
        """
        
        # If we have a valid OpenAI API key, use it for general questions
        if self.client.api_key and self.client.api_key != "your_openai_api_key_here":
            # Define personality-specific prompts
            personality_prompts = {
                "enthusiastic": "You are an incredibly excited Pokémon expert who LOVES talking about Pokémon! Use lots of exclamation points and energetic language!",
                "scholarly": "You are a distinguished Pokémon researcher who speaks in an academic, detailed manner with scientific precision.",
                "casual": "You are a chill Pokémon fan who talks like you're chatting with a friend. Keep it relaxed and conversational.",
                "factual": "You are a Pokémon database that provides clear, concise, factual information without embellishment.",
                "random": "You are a passionate Pokémon expert with a fun, engaging personality. Vary your response style randomly between enthusiastic, scholarly, and casual approaches."
            }
            
            # Use the specified personality or default to random
            system_prompt = personality_prompts.get(personality_mode, personality_prompts["random"])
            system_prompt += """ Answer questions about general Pokémon knowledge, including types, regions, mechanics, games, and lore. Make your responses engaging while staying factually accurate."""
            
            user_prompt = f"""Question: {question}

Please provide a helpful answer about Pokémon. This is a general knowledge question, not about a specific Pokémon.

Some key Pokémon facts you can reference:
- There are 18 different Pokémon types: Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, and Fairy
- Pokémon games are set in various regions like Kanto, Johto, Hoenn, Sinnoh, Unova, Kalos, Alola, and Galar
- The franchise includes games, anime, trading cards, and more
- Pokémon can evolve, learn moves, and have abilities
- There are legendary and mythical Pokémon that are rare and powerful

Please answer this question with accurate information!"""
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error generating answer: {e}"
        
        # Otherwise, use rule-based answers for common general questions
        question_lower = question.lower()
        
        if "how many" in question_lower and "type" in question_lower:
            return "There are 18 different Pokémon types: Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, and Fairy."
            
        elif "what are" in question_lower and "type" in question_lower:
            return "The 18 Pokémon types are: Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, and Fairy. Each type has strengths and weaknesses against other types."
            
        elif "region" in question_lower or "regions" in question_lower:
            return "Pokémon games are set in various regions including Kanto, Johto, Hoenn, Sinnoh, Unova, Kalos, Alola, and Galar. Each region has its own unique Pokémon, gym leaders, and storylines."
            
        elif "generation" in question_lower or "generations" in question_lower:
            return "There are currently 9 generations of Pokémon games, starting with Generation I (Red/Blue) and continuing through Generation IX (Scarlet/Violet). Each generation typically introduces new Pokémon, mechanics, and regions."
            
        elif "evolution" in question_lower and not any(pokemon in question_lower for pokemon in ['charizard', 'pikachu', 'eevee']):
            return "Pokémon evolution is the process by which certain Pokémon transform into different species. Evolution can be triggered by leveling up, using items, trading, friendship, or other special conditions."
            
        elif "legendary" in question_lower or "mythical" in question_lower:
            return "Legendary and Mythical Pokémon are rare, powerful creatures with unique roles in Pokémon lore. Examples include Mewtwo, Mew, Articuno, Zapdos, Moltres, and many others from various generations."
            
        elif "starter" in question_lower or "starters" in question_lower:
            return "Starter Pokémon are the first Pokémon given to new trainers. They typically come in sets of three representing Fire, Water, and Grass types. Famous examples include Bulbasaur, Charmander, and Squirtle from Generation I."
            
        elif "shiny" in question_lower:
            return "Shiny Pokémon are rare variants with different color schemes from their normal counterparts. They have the same stats but are highly sought after by collectors due to their rarity."
            
        elif "ability" in question_lower or "abilities" in question_lower:
            return "Pokémon abilities are special traits that can affect battle, exploration, or other aspects of gameplay. Each Pokémon species has one or more possible abilities, and some have hidden abilities."
            
        elif "gym" in question_lower or "elite four" in question_lower or "champion" in question_lower:
            return "Gym Leaders are specialized trainers who focus on specific Pokémon types. After defeating 8 Gym Leaders, trainers typically face the Elite Four and Champion to become the regional champion."
            
        # Default response for general questions
        return "I'm a Pokémon expert! I can answer questions about Pokémon types, regions, evolution, legendary Pokémon, and much more. For specific Pokémon information, just mention the Pokémon's name in your question!"

# Example usage:
# rag_system = RAGSystem(llm_api_key="your_openai_api_key")
# questions = ["What type is Charizard?", "How much does Pikachu weigh?"]
# ground_truth_answers = {"What type is Charizard?": "Fire/Flying", "How much does Pikachu weigh?": "6 kg"}

# Basic evaluation
# results = rag_system.evaluate_answers(questions, ground_truth_answers)

# Test different personalities
# enthusiastic_results = rag_system.evaluate_answers(questions, ground_truth_answers, personality_mode="enthusiastic")
# scholarly_results = rag_system.evaluate_answers(questions, ground_truth_answers, personality_mode="scholarly")

# Test response variety
# variety_results = rag_system.evaluate_answers(questions, ground_truth_answers, test_variety=True)
# print(variety_results)