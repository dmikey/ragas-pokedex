# Pokémon Retrieval-Augmented Generation QA System 🔥 Powered by RAGAS

This project implements a Retrieval-Augmented Generation (RAG) Question-Answering (QA) system using the PokéAPI. The application accepts Pokémon-related questions, retrieves answers from the PokéAPI, generates natural-language responses using a language model, and evaluates the results using **RAGAS (Retrieval-Augmented Generation Assessment)** metrics for comprehensive evaluation of answer relevancy, faithfulness, correctness, and similarity.

## Project Structure

```
pokemon-rag-qa
├── src
│   ├── app.py             # Flask web application
│   ├── pokemon_api.py     # Functions to interact with the PokéAPI
│   ├── rag_system.py      # RAG logic for generating answers
│   ├── ragas_evaluator.py # RAGAS evaluation integration
│   ├── evaluator.py       # Legacy evaluation (kept for fallback)
│   └── templates/
│       └── index.html     # Web interface template
├── demo_ragas.py          # Demo script showing RAGAS integration
├── requirements.txt       # Project dependencies
├── .env.example          # Example environment variables
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/pokemon-rag-qa.git
   cd pokemon-rag-qa
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   
   Copy the `.env.example` to `.env` and fill in the necessary API keys and configuration settings:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the Application**
   
   Execute the Flask web application:
   ```bash
   python src/app.py
   ```
   
   Then open your browser to `http://localhost:5000`

## Features

### 🔥 RAGAS Integration
This application now uses **RAGAS (Retrieval-Augmented Generation Assessment)** for comprehensive evaluation:

- **Answer Relevancy**: Measures how well the answer addresses the question
- **Faithfulness**: Evaluates if the answer is grounded in the provided context
- **Answer Correctness**: Assesses the factual accuracy of the response
- **Answer Similarity**: Compares semantic similarity between responses

### 🎮 Interactive Web Interface
- Real-time Pokemon question answering
- RAGAS evaluation scores displayed for each response
- Variety testing with multiple response generation
- Multiple personality modes (factual, friendly, expert, casual)

### 📊 Comprehensive Evaluation
- Single answer evaluation with RAGAS metrics
- Variety analysis for response diversity
- Fallback scoring when RAGAS evaluation fails
- Detailed scoring breakdown for transparency

## Usage

### Web Interface
1. Start the Flask application: `python src/app.py`
2. Open your browser to `http://localhost:5000`
3. Ask Pokemon-related questions in the chat interface
4. View RAGAS evaluation scores for each response
5. Test response variety with the "Variety Test" feature

### Demo Script
Run the RAGAS integration demo:
```bash
python demo_ragas.py
```

## Example Questions

- "What type is Charizard?"
- "How much does Pikachu weigh?"
- "What are Bulbasaur's abilities?"
- "What moves can Squirtle learn?"

## RAGAS Evaluation Metrics

The system evaluates responses using these RAGAS metrics:

- **Factual Accuracy**: Based on answer_correctness and faithfulness
- **Response Quality**: Based on answer_relevancy  
- **Engagement**: Derived from response characteristics
- **Similarity**: Using answer_similarity for variety testing
- **Overall Score**: Weighted combination of all metrics

## Dependencies

Key dependencies include:
- `ragas`: For RAG evaluation metrics
- `flask`: Web framework
- `openai`: For language model integration
- `requests`: For PokéAPI calls
- `datasets`: For RAGAS dataset creation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.