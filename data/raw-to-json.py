import sys
import json

def main():
    """
        Take raw data from newline terminated file, convert to JSON and write to file.
    """
    file_name = sys.argv[1]
    output_name = sys.argv[2]
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
    with open(output_name, 'a+') as f:
        # Check if file exists
        try:
            data = json.load(f)
            for line in lines:
                data[line] = None
        except ValueError:
            # If not, create it
            data = {}
            for line in lines:
                data[line] = None
            json.dump(data, f)

if __name__ == '__main__':
    main()