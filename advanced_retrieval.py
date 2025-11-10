import json
from sentence_transformers import SentenceTransformer
import chromadb

with open('courses.json', 'r') as f:
    courses = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
documents = []
metadatas = []
ids = []
id = 0

for course in courses:
  doc_text = f"Course: {course['title']} : {course['course']}. Description: {course['description']}."
  documents.append(doc_text)
  metadatas.append(course)
  ids.append(str(id))
  id += 1

embeddings = model.encode(documents, show_progress_bar=True)

db_path = './rice_courses_db'
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection("rice_courses")

# Set a batch size safely under the 5461 limit. 5000 is a safe, round number.
batch_size = 5000
total_items = len(ids)
num_batches = ((total_items + batch_size - 1) // batch_size) #ceil of division

print(f"Total items: {total_items}. Batch size: {batch_size}. Sending in {num_batches} batches...")

for i in range(0, total_items, batch_size):

    # Calculate the end index for the current batch
    end_index = min(i + batch_size, total_items)

    print(f"Adding batch {i // batch_size + 1}/{num_batches} (items {i} to {end_index})...")

    # Get the sublists
    batch_embeddings = embeddings[i:end_index]
    batch_documents = documents[i:end_index]
    batch_metadatas = metadatas[i:end_index]
    batch_ids = ids[i:end_index]

    collection.add(
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas,
        ids=batch_ids
        )

