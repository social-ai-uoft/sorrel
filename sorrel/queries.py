import ast

import google.genai

# TODO: self-correcting process

JSON_PROMPT = "You are a configuration assistant for the Sorrel MARL framework. Your task is to analyze the user's request and populate the following JSON schema with corresponding values. You must extract all parameters explicitly mentioned by the user. For any parameters that are not mentioned, you must infer reasonable default values appropriate for a typical MARL experiment. The JSON output must be valid and adhere strictly to the provided schema."

JSON_SCHEMA = ""

ENTITY_PROMPT = "You are an expert Python programmer specializing in the Sorrel MARL framework. Using the provided context about the framework's base classes and code examples, write a complete Python class for the '{entity_type}' entity. The class must inherit from sorrel.components.base.Entity. Implement any attributes or methods described in the user's original request, such as a transition method for dynamic behavior."

ENVIRONMENT_PROMPT = "You are an expert Python programmer and systems designer for the Sorrel MARL framework. Your task is to write a complete environment.py file that implements the experiment described by the user and specified in the provided configuration. The main class must inherit from sorrel.environment.Environment. You MUST correctly implement the abstract methods setup_agents() and populate_environment(). The logic in these methods must use the parameters from the provided config.json file to set up the world."

ERROR_PROMPT = "The previous code you generated was syntactically incorrect and resulted in the following error: {error_message}. Please analyze the error and the provided code, and generate a corrected version of the complete file."


def prompt(input: str, client: google.genai.Client, directory: str, model: str):
    response = client.models.generate_content(model=model, contents=[])

    # ask where directory

    # Process user input into json config

    # Generate entities

    # Generate environment


def verify(code: str, prev_prompts: list[str], client: google.genai.Client, model: str):
    code = code
    compiles = False

    attempts = 0
    while not compiles and attempts < 10:
        try:
            ast.parse(code)
            compiles = True
        except SyntaxError as e:
            response = client.models.generate_content(
                model=model, contents=prev_prompts.append(ERROR_PROMPT)
            )
            code = response.text
            attempts += 1
    if not compiles:
        raise SyntaxError("Failed to generate valid code after multiple attempts.")


if __name__ == "__main__":
    client = google.genai.Client(
        vertexai=True, api_key="AIzaSyAvoQSQznBdXBEKrRfDvGA82IJDG6eGVdM"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=["How does AI work?"]
    )
    print(response.text)
