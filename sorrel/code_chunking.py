import os

import chromadb
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# TODO: change the embedding model


def chunk_doc_file(file_path: str, collection: chromadb.Collection):
    """Chunk a markdown file into meaningful parts."""


def chunk_code_file(file_path: str, collection: chromadb.Collection):
    """Chunk a Python code file into meaningful parts.

    IDs: 'file_path:chunk_number (indexed from 0)'

    Metadata fields:
        file_path
        is_import
        is_class
        class_name
        is_function
        function_name
        is_class_function (if so, also provide class name)
    """

    ids = []
    documents = []
    metadatas = []  # list of dicts

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)

    print(f"Chunking file: {file_path}")

    with open(file_path, "rb") as f:
        code = f.read()
        tree = parser.parse(code)
        assert tree.root_node.type == "module", "Root node is not a module."

        module_node = tree.root_node
        end_byte = module_node.end_byte

        if not module_node.children:
            # Empty module
            ids.append(f"{file_path}:0")
            documents.append(code.decode("utf-8"))
            metadatas.append({"file_path": file_path})

        # Chunk that includes all imports (i.e. before the first class or function definition)
        i = 0
        while i < len(module_node.children) and module_node.children[i].type not in [
            "class_definition",
            "function_definition",
        ]:
            i += 1
        if i > 0:
            end_byte = module_node.children[i - 1].end_byte
            ids.append(f"{file_path}:0")
            documents.append(code[:end_byte].decode("utf-8"))
            metadatas.append(
                {
                    "file_path": file_path,
                    "is_import": True,
                    "is_class": False,
                    "is_function": False,
                }
            )

        # Process each class and function definition
        j = i
        start_byte = end_byte if i > 0 else 0
        while j < len(module_node.children):
            child = module_node.children[j]

            if child.type == "class_definition":
                # TODO: if a class chunk is too long, split into class function chunks;
                # ex. Gemini's embedding model has a limit of 2048 tokens
                class_name = child.child_by_field_name("name").text.decode("utf-8")
                end_byte = child.end_byte
                ids.append(f"{file_path}:{len(ids)}")
                documents.append(code[start_byte:end_byte].decode("utf-8"))
                metadatas.append(
                    {
                        "file_path": file_path,
                        "is_import": False,
                        "is_class": True,
                        "class_name": class_name,
                        "is_function": False,
                    }
                )
                start_byte = end_byte

            elif child.type == "function_definition":
                function_name = child.child_by_field_name("name").text.decode("utf-8")
                end_byte = child.end_byte
                ids.append(f"{file_path}:{len(ids)}")
                documents.append(code[start_byte:end_byte].decode("utf-8"))
                metadatas.append(
                    {
                        "file_path": file_path,
                        "is_import": False,
                        "is_class": False,
                        "is_function": True,
                        "function_name": function_name,
                        "is_class_function": False,
                    }
                )
                start_byte = end_byte
            j += 1

        # Add chunk of any remaining code after the last class or function definition (comments, etc.)
        # - 2 to truncate the new line at the end of every python file (assuming CRLF and UTF-8)
        if start_byte < len(code) - 2:
            ids.append(f"{file_path}:{len(ids)}")
            documents.append(code[start_byte:].decode("utf-8"))
            metadatas.append(
                {
                    "file_path": file_path,
                    "is_import": False,
                    "is_class": False,
                    "is_function": False,
                }
            )

    # Upsert to ChromaDB
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # As part of updating, remove extraneous previously existing chunks for this file
    k = len(ids)
    while collection.get(ids=[f"{file_path}:{k}"], include=[])[
        "ids"
    ]:  # only returns ids
        collection.delete(ids=[f"{file_path}:{k}"])
        k += 1


def chunk_code_directory(directory_path: str, collection: chromadb.Collection):
    """Chunk all Python files in a directory."""

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                chunk_code_file(file_path, collection)

    print(f"Finished processing directory: {directory_path}")
    print(f"Total chunks in collection: {collection.count()}")


def test_retrieval(collection: chromadb.Collection):
    """Test retrieval of chunks from the collection."""
    query_text = "Python class for a Sorrel Entity named Apple that can be consumed and respawns randomly."
    print(f"Querying for: {query_text}")
    print("-" * 40)
    results = collection.query(
        query_texts=[query_text],
        n_results=5,
    )

    print("Retrieved results:")
    for doc, met, dis in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"Document: {doc}")
        print(f"Metadata: {met}")
        print(f"Distance: {dis}")
        print("-" * 40)


if __name__ == "__main__":

    chroma_client = chromadb.PersistentClient(path="/chroma_db")
    collection = chroma_client.get_or_create_collection(name="test_collection")

    chunk_code_directory("sorrel/", collection)

    test_retrieval(collection)

    # Clean up the collection after processing
    # chroma_client.delete_collection(name="test_collection")
    # chroma_client.reset()

    # types:
    # module
    # comment
    # future_import_statement
    # import_statement
    # import_from_statement
    # class_definition
    #   class, identifier, :, block
    #       expression_statement, function_definition
    # function_definition
    #   def, identifier, parameters, ->, type, :, block

    # field_names (for functions):
    # name (functions, classes, etc)
    # parameters
    # return_type
    # body
