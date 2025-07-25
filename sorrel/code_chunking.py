import tree_sitter_python as tspython
from tree_sitter import Language, Parser

if __name__ == "__main__":
    PY_LANGUAGE = Language(tspython.language())

    parser = Parser(PY_LANGUAGE)

    with open("sorrel/buffers.py", "rb") as f:
        code = f.read()
        tree = parser.parse(code)
        print(tree.root_node.type)
        child = tree.root_node.children[0]
        print([c.type for c in tree.root_node.children])
        print(child.type, child.start_byte, child.end_byte)
        print(code[child.start_byte : child.end_byte].decode("utf-8"))
