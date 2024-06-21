import ollama




def main():
    """
    Test using guidance together with ollama.
    """
    ollama.pull('llama3')
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ])
    print(response['message']['content'])



if __name__ == '__main__':
    main()