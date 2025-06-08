# Configuration settings for the Pokémon RAG QA system

# API endpoint for the PokéAPI
POKEAPI_URL = "https://pokeapi.co/api/v2"

# Model parameters for the LLM
LLM_MODEL_NAME = "gpt-3.5-turbo"  # Example model name, adjust as necessary
LLM_MAX_TOKENS = 150  # Maximum tokens for the generated response
LLM_TEMPERATURE = 0.7  # Controls the randomness of the output

# Evaluation settings
EVALUATION_METRICS = ["correctness", "faithfulness"]  # Metrics to evaluate

# Other constants
SAMPLE_QUESTIONS_FILE = "data/sample_questions.json"
EVALUATION_RESULTS_FILE = "data/evaluation_results.json"