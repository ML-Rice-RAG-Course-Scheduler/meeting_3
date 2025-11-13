import chromadb
import os
from ollama import Client
from advanced_retrieval import expanded_retrieve

client = chromadb.PersistentClient(path=os.path.abspath("rice_courses_db"))
collection = client.get_collection("rice_courses")


ollama = Client(host="http://localhost:11434")

def generate_answer(user_query):
    _, results = expanded_retrieve(user_query, top_k=5)
    context = "\n\n".join([
        f"{r['meta']['course']}: {r['meta']['title']}\n{r['doc']}"
        for r in results
    ])
    response = ollama.chat(
        model='llama3.2',
        messages=[{
            'role': 'system',
            'content': f"You are university course scheduling assistant. Your job is to answer the user's question based on the provided course data.\n\nContext: {context}"
        }, {
            'role': 'user',
            'content': user_query
        }]
    )
    return response['message']['content']


print(generate_answer("What courses are about African religions?"))
print("\nClasses: ")
print(answer)