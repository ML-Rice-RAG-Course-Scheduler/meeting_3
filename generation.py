import chromadb
import os

client = chromadb.PersistentClient(path=os.path.abspath("rice_courses_db"))
collection = client.get_collection("rice_courses")