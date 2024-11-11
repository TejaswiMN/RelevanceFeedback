import tkinter as tk
from tkinter import messagebox, ttk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
# import networkx as nx

# Retrieve documents from Arxiv API
def retrieve_documents(query, k=5):
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    response = requests.get(ARXIV_API_URL, params=params)
    if response.status_code != 200:
        messagebox.showerror("Error", "Failed to retrieve documents from Arxiv.")
        return []
    
    entries = response.text.split("<entry>")
    papers = []
    for entry in entries[1:]:
        title = entry.split("<title>")[1].split("</title>")[0].strip()
        summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
        authors = [a.split("</name>")[0] for a in entry.split("<name>")[1:]]
        paper_text = f"{title}\nAuthors: {', '.join(authors)}\nSummary: {summary}"
        papers.append((title, summary, paper_text))  # Return both title and summary
    return papers[:k]

# Function to process feedback and refine query
def update_documents_with_feedback(query, relevant_docs, k):
    expanded_query = f"{query} {' '.join(relevant_docs)}"
    return retrieve_documents(expanded_query, k)

# Calculate Precision and Recall
def calculate_precision_recall(retrieved_docs, relevant_docs, all_relevant_docs):
    # Precision: how many retrieved docs are relevant
    retrieved_relevant = [doc for doc in retrieved_docs if doc in relevant_docs]
    precision = len(retrieved_relevant) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0

    # Recall: how many relevant docs were retrieved
    recall = len(retrieved_relevant) / len(all_relevant_docs) if len(all_relevant_docs) > 0 else 0

    return precision, recall

# GUI Application for Search Engine
class RelevanceFeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Search Engine with Relevance Feedback")
        self.root.geometry("700x800")

        # Main Frame
        main_frame = tk.Frame(root, padx=20, pady=20, bg="#1c1e21")
        main_frame.pack(fill="both", expand=True)

        # Title Label
        title_label = tk.Label(main_frame, text="Document Retrieval System", font=("Helvetica", 18, "bold"), bg="#1c1e21", fg="white")
        title_label.pack(pady=10)

        # Query Frame
        query_frame = tk.Frame(main_frame, bg="#1c1e21")
        query_frame.pack(pady=10)
        self.query_label = tk.Label(query_frame, text="Enter your query:", bg="#1c1e21", fg="white", font=("Helvetica", 12))
        self.query_label.grid(row=0, column=0, sticky="w")
        self.query_entry = tk.Entry(query_frame, width=45, font=("Helvetica", 12))
        self.query_entry.grid(row=1, column=0, padx=5, pady=5)
        self.search_button = tk.Button(query_frame, text="Search", command=self.search_documents, font=("Helvetica", 12), bg="#4CAF50", fg="white")
        self.search_button.grid(row=1, column=1, padx=5)

        # Results Frame with Scrollbar
        results_frame = tk.Frame(main_frame, bg="#1c1e21")
        results_frame.pack(pady=10, fill="both", expand=True)
        self.canvas = tk.Canvas(results_frame, bg="#1c1e21", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.result_frame = tk.Frame(self.canvas, bg="#1c1e21")
        self.canvas.create_window((0, 0), window=self.result_frame, anchor="nw")

        # Variables
        self.k = 5
        self.query = ""
        self.feedback = []
        self.all_relevant_docs = []  # Store all relevant docs for recall calculation

    def search_documents(self):
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        self.display_documents(self.query)

    def display_documents(self, query):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.feedback = []

        # Retrieve documents from API
        top_docs = retrieve_documents(query, self.k)
        if not top_docs:
            messagebox.showinfo("No Results", "No documents found.")
            return

        # Vectorize and rank documents using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([query] + [doc[2] for doc in top_docs])  # Use full text (title + summary)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked_docs = sorted(zip(top_docs, cosine_similarities), key=lambda x: x[1], reverse=True)

        # Display documents with checkboxes for relevance feedback
        self.doc_labels = []
        self.relevance_checkbuttons = []

        for i, ((title, summary, paper_text), score) in enumerate(ranked_docs):
            doc_label = tk.Label(self.result_frame, text=f"Document {i+1}:\n{title}\n(Score: {score:.3f})", wraplength=500, bg="#333", fg="white", font=("Helvetica", 10), justify="left", anchor="w")
            doc_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            feedback_var = tk.StringVar(value="Irrelevant")  # Default feedback to 'Irrelevant'
            feedback_cb = ttk.Combobox(self.result_frame, textvariable=feedback_var, values=["Highly Relevant", "Relevant", "Irrelevant"], state="readonly", width=15)
            feedback_cb.grid(row=i, column=1, padx=10, pady=10)

            self.doc_labels.append((title, summary))  # Store title and summary
            self.relevance_checkbuttons.append(feedback_var)

        # Submit Feedback Button
        self.feedback_button = tk.Button(self.result_frame, text="Submit Feedback", command=self.submit_feedback, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.feedback_button.grid(row=len(ranked_docs), column=0, columnspan=2, pady=10)

    def submit_feedback(self):
        feedback = [var.get() for var in self.relevance_checkbuttons]
        relevant_docs = [self.doc_labels[i][0] for i, fb in enumerate(feedback) if fb != "Irrelevant"]

        if not relevant_docs:
            messagebox.showwarning("Feedback Error", "Please select at least one relevant document.")
            return

        expanded_query = f"{self.query} {' '.join(relevant_docs)}"
        updated_docs = update_documents_with_feedback(expanded_query, relevant_docs, self.k)

        # Store relevant docs for recall evaluation
        self.all_relevant_docs.extend(relevant_docs)

        # Calculate Precision and Recall
        # precision, recall = calculate_precision_recall(updated_docs, relevant_docs, self.all_relevant_docs)
        # messagebox.showinfo("Evaluation", f"Precision: {precision:.2f}, Recall: {recall:.2f}")

        # Display updated documents
        self.display_documents(expanded_query)

# Main Application Window
if __name__ == "__main__":
    root = tk.Tk()
    app = RelevanceFeedbackApp(root)
    root.mainloop()
