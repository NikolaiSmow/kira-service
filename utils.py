def get_number_of_words(content: str):
    if content:
        return len(content.split(" ")) - 1
    else:
        return 0


def get_number_of_characters(content: str):
    if content:
        return len(content)
    else:
        return 0


def convert_lines_to_characters(num_lines: int, num_characters_per_line: int):
    return num_lines * num_characters_per_line


def convert_characters_to_lines(num_characters: int, num_characters_per_line: int):
    return num_characters // num_characters_per_line
