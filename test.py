import ollama

response = ollama.chat(
    model= 'llama3.1',
    messages= [
        {'role': 'user', 'content': 'What is diabetes'}
    ]
)

print(response['message']['content'])