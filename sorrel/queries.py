import ast
import json
from pathlib import Path

import chromadb

# TODO: maybe pip uninstall google-genais
import google.genai
from google.genai.types import GenerateContentConfig

TESTING = True  # Set to True to enable debug prints

# ChromaDB configs
NUM_RESULTS = 5

# Gemini configs
MODEL = "gemini-2.5-pro"

JSON_SYSTEM_INST = "You are a configuration assistant for the Sorrel MARL framework. Your task is to analyze the user's request and populate the following JSON schema with corresponding values. You must extract all parameters explicitly mentioned by the user. For any parameters that are not mentioned, you must infer reasonable default values appropriate for a typical MARL experiment. The JSON output must be valid and adhere strictly to the provided schema."

JSON_SCHEMA = '{"experiment": {"name": "", "epochs": 1000, "max_turns": 200, "record_period": 100}, "world": {"class": "GridWorld", "width": 30, "height": 30, "default_entity": "EmptyEntity"}, "agents": [], "entities": []}'

ENTITY_PROMPT = "You are an expert Python programmer specializing in the Sorrel MARL framework. Using the provided context about the framework's base classes and code examples, write a complete Python class for the entity with the following configs: {entity_configs}. The class must inherit from sorrel.entities.Entity. Implement any attributes or methods described in the user's original request, such as a transition method for dynamic behavior. Only generate the contents of the code file, without any additional explanations or comments. Ensure that the code is valid Python."

AGENT_PROMPT = "You are an expert Python programmer specializing in the Sorrel MARL framework. Using the provided context about the framework's base classes and code examples, write a complete Python class for the agent with the following configs: {agent_configs}. The class must inherit from sorrel.agents.Agent. Implement any attributes or methods described in the user's original request, such as a transition method for dynamic behavior. Only generate the contents of the code file, without any additional explanations or comments. Ensure that the code is valid Python."

ENVIRONMENT_PROMPT = "You are an expert Python programmer and systems designer for the Sorrel MARL framework. Your task is to write a complete environment.py file that implements the experiment described by the user and specified in the provided configuration. The main class must inherit from sorrel.worlds.Gridworld. You MUST correctly implement the abstract methods setup_agents() and populate_environment(). The logic in these methods must use the parameters from the provided config.json file to set up the world. Only generate the contents of the code file, without any additional explanations or comments. Ensure that the code is valid Python."

ERROR_PROMPT = "The previous code you generated was syntactically incorrect and resulted in the following error: {error_message}. Please analyze the error and the provided code, and generate a corrected version of the complete file. Only generate the contents of the code file, without any additional explanations or comments. Ensure that the code is valid Python."

MAX_ATTEMPTS = 3  # Max attempts to fix syntax errors in generated code


def query_database(
    query: str, num_results: int, collection: chromadb.Collection
) -> list[str]:
    """Query the ChromaDB database for relevant code snippets, and attach all relevant
    import statements."""
    # Only search for either documents or non-import code snippets
    results = collection.query(
        query_texts=[query],
        n_results=num_results,
        where={"$or": [{"is_code": False}, {"is_import": False}]},
    )
    completed_results = []
    metadatas = results["metadatas"][0]
    # Attach imports to code snippets
    for i in range(num_results):
        if metadatas[i]["is_code"]:
            file_path = metadatas[i]["file_path"]
            imports = collection.get(
                where={"$and": [{"file_path": file_path}, {"is_import": True}]},
                include=["documents"],
            )["documents"][0]
            completed_code = imports + results["documents"][0][i]
            completed_results.append(completed_code)
        else:
            completed_results.append(results["documents"][0][i])
    if TESTING:
        print(f"Query to database: {query}")
        print(f"Database query results:")
        for i, result in enumerate(completed_results):
            print(f"Result {i+1}:\n{result[:100]}...")
    return completed_results


def prompt(
    google_client: google.genai.Client,
    collection: chromadb.Collection,
    interactive: bool = True,
    output_dir: str | Path | None = None,
    request: str | None = None,
) -> None:

    # Get inputs
    if not interactive:
        if output_dir is None:
            raise ValueError(
                "Output directory must be specified in non-interactive mode."
            )
        if request is None:
            raise ValueError("Request must be specified in non-interactive mode.")
    else:
        output_dir = input("Enter the directory to save the generated files: ")
        request = input(
            "Please describe the Sorrel MARL experiment you wish to create: "
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process user input into json config
    json_response = google_client.models.generate_content(
        model=MODEL,
        config=GenerateContentConfig(
            system_instruction=[JSON_SYSTEM_INST, JSON_SCHEMA]
        ),
        contents="User request: " + request,
    )

    # Gemini outputs with backticks, so we remove them
    json_response_text = json_response.text.replace("```json", "").replace("```", "")
    print("Inferred json configs:", json_response_text)

    json_configs = json.loads(json_response_text)

    # Generate entities
    # TODO: entities generated in seperate files might cause problems with imports
    entities = json_configs.get("entities", [])
    entity_codes = []
    for i, entity in enumerate(entities):
        query_results = query_database(
            query=f"Classes and tutorials related to an entity with the following configs: {entity}",
            num_results=NUM_RESULTS,
            collection=collection,
        )
        prompts = [ENTITY_PROMPT.format(entity_configs=entity)] + query_results
        entity_response = google_client.models.generate_content(
            model=MODEL,
            contents=prompts,
        )
        entity_code = entity_response.text.replace("```python", "").replace("```", "")
        verified_code = verify(entity_code, prompts, google_client, MODEL)
        file_name = entity.get("class", f"Entity{i}").lower() + ".py"
        with open(output_dir / file_name, "w+") as f:
            f.write(verified_code)
        file_path = output_dir / file_name
        entity_codes.append(
            "The following code snippet comes from the file"
            + file_path
            + ": \n"
            + verified_code
        )

    # Generate agents
    # TODO
    agents = json_configs.get("agents", [])
    agent_codes = []
    for i, agent in enumerate(agents):
        query_results = query_database(
            query=f"Classes and tutorials related to an agent with the following configs: {agent}",
            num_results=NUM_RESULTS,
            collection=collection,
        )
        prompts = [AGENT_PROMPT.format(agent_configs=agent)] + query_results
        agent_response = google_client.models.generate_content(
            model=MODEL,
            contents=prompts,
        )
        agent_code = agent_response.text.replace("```python", "").replace("```", "")
        verified_code = verify(agent_code, prompts, google_client, MODEL)
        file_name = agent.get("class", f"Agent{i}").lower() + ".py"
        with open(output_dir / file_name, "w+") as f:
            f.write(verified_code)
        agent_codes.append(
            "The following code snippet comes from the file"
            + file_path
            + ": \n"
            + verified_code
        )

    # Generate environment
    query_results = query_database(
        query=f"Classes and tutorials related to a sorrel environment implementing the following experiment: {json_configs}",
        num_results=NUM_RESULTS,
        collection=collection,
    )
    prompts = [ENVIRONMENT_PROMPT] + query_results + entity_codes + agent_codes
    env_response = google_client.models.generate_content(
        model=MODEL,
        contents=prompts,
    )
    env_code = env_response.text.replace("```python", "").replace("```", "")
    verified_code = verify(env_code, prompts, google_client, MODEL)
    with open(output_dir / "environment.py", "w+") as f:
        f.write(verified_code)


def verify(
    generated_code: str,
    prev_prompts: list[str],
    client: google.genai.Client,
    model: str,
) -> str:
    code = generated_code
    compiles = False

    attempts = 0
    while not compiles and attempts < MAX_ATTEMPTS:
        try:
            ast.parse(code)
            compiles = True
        except SyntaxError as e:
            print("Syntax error detected, regenerating code...")
            response = client.models.generate_content(
                model=model, contents=[code, ERROR_PROMPT.format(error_message=str(e))]
            )
            code = response.text.replace("```python", "").replace("```", "")
            attempts += 1
    if not compiles:
        print("Failed code: ", code)
        raise SyntaxError("Failed to generate valid code after multiple attempts.")

    return code


if __name__ == "__main__":

    google_client = google.genai.Client(
        vertexai=True,
        project="sorrel-gemini",
        location="us-central1",
    )
    chroma_client = chromadb.PersistentClient(path="/chroma_db")
    collection_name = "test_collection"
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # prompt(google_client, collection, interactive=False, output_dir="test-rag/", request="Create a sorrel environment based on the 'Cleanup' scenario. It should be a 15x15 grid world. The world has a river running down the middle that starts clean but gets polluted when agents are nearby. There should be 7 agents who are rewarded for cleaning the river by firing a cleaning beam, but this action has a small cost. They are also incentivized to eat apples that spawn in the world. Apples should respawn randomly after being eaten.")
    query_database(
        "Classes and tutorials related to an entity with the following configs: An entity representing a river that can become polluted over time when agents are nearby. The river should have a transition method that adds pollution entities to its location based on certain conditions.",
        num_results=5,
        collection=collection,
    )
