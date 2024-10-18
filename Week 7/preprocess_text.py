def read_file_into_list(file_path):
    """
    Reads a text file and returns a list where each line is a list item.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Strip newline characters from each line
        lines = [line.strip() for line in lines]
    return lines

def main(file_path):
    return read_file_into_list(file_path)
