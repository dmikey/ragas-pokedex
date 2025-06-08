import requests

POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"

def get_pokemon_data(pokemon_name):
    """Fetches data for a given Pokémon from the PokéAPI."""
    response = requests.get(f"{POKEAPI_BASE_URL}/pokemon/{pokemon_name.lower()}")
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Pokémon '{pokemon_name}' not found.")

def get_pokemon_type(pokemon_name):
    """Retrieves the type of a given Pokémon."""
    data = get_pokemon_data(pokemon_name)
    types = [type_info['type']['name'] for type_info in data['types']]
    return types

def get_pokemon_weight(pokemon_name):
    """Retrieves the weight of a given Pokémon."""
    data = get_pokemon_data(pokemon_name)
    weight = data['weight']  # Weight is in hectograms
    return weight

def get_pokemon_height(pokemon_name):
    """Retrieves the height of a given Pokémon."""
    data = get_pokemon_data(pokemon_name)
    height = data['height']  # Height is in decimeters
    return height

def get_pokemon_abilities(pokemon_name):
    """Retrieves the abilities of a given Pokémon."""
    data = get_pokemon_data(pokemon_name)
    abilities = [ability_info['ability']['name'] for ability_info in data['abilities']]
    return abilities

def get_pokemon_info(pokemon_name):
    """Retrieves various information about a given Pokémon."""
    data = get_pokemon_data(pokemon_name)
    return {
        "name": data['name'],
        "types": get_pokemon_type(pokemon_name),
        "weight": get_pokemon_weight(pokemon_name),
        "height": get_pokemon_height(pokemon_name),
        "abilities": get_pokemon_abilities(pokemon_name)
    }

# Alias for compatibility with main.py and rag_system.py
def fetch_pokemon_data(pokemon_name):
    """Fetches data for a given Pokémon. Alias for get_pokemon_info."""
    return get_pokemon_info(pokemon_name)