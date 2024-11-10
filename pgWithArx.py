import tkinter as tk
from tkinter import messagebox, ttk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to retrieve documents from the Arxiv API
def retrieve_documents(query, k=3):
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
        papers.append((paper_text, 1.0))  # Placeholder score
    return papers[:k]

# Function to update documents based on feedback
def update_documents_with_feedback(query, relevant_docs, k):
    expanded_query = ' '.join(relevant_docs)
    return retrieve_documents(expanded_query, k)

# GUI Application
class RelevanceFeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Retrieval with Relevance Feedback")
        self.root.geometry("600x700")

        # Bind resize event
        self.root.bind("<Configure>", self.resize)

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
        self.result_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.feedback = []
        self.k = 5 # Number of top documents to retrieve
        self.query = ""  # Current query text

    def search_documents(self):
        self.query = self.query_entry.get().strip()
        self.original_query = self.query  # Save original query
        if not self.query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        self.display_documents()

    def resize(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def display_documents(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        self.doc_labels = []
        self.relevance_checkbuttons = []
        
        top_docs = retrieve_documents(self.query, self.k)
        self.feedback = []

        for i, (doc, score) in enumerate(top_docs):
            doc_label = tk.Label(self.result_frame, text=f"Document {i+1}: {doc}\n(Score: {score:.3f})", wraplength=450, bg="#333", fg="white", font=("Helvetica", 10), justify="left", anchor="w")
            doc_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            feedback_var = tk.IntVar(value=0)
            feedback_cb = tk.Checkbutton(self.result_frame, text="Relevant", variable=feedback_var, bg="#1c1e21", fg="white", selectcolor="#333")
            feedback_cb.grid(row=i, column=1, padx=10, pady=10)

            self.doc_labels.append(doc_label)
            self.relevance_checkbuttons.append(feedback_var)

        # Submit Feedback Button
        self.feedback_button = tk.Button(self.result_frame, text="Submit Feedback", command=self.submit_feedback, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.feedback_button.grid(row=self.k, column=0, columnspan=2, pady=10)

    def submit_feedback(self):
        feedback = [var.get() for var in self.relevance_checkbuttons]
        relevant_docs = [self.doc_labels[i].cget("text").split("\n")[0] for i, is_relevant in enumerate(feedback) if is_relevant] # Only title

        if not relevant_docs:
            messagebox.showwarning("Feedback Error", "Please select at least one relevant document.")
            return
        expanded_query = f"{self.original_query} {' '.join(relevant_docs)}"
        updated_docs = update_documents_with_feedback(expanded_query, relevant_docs, self.k)
        self.query = expanded_query
        self.display_updated_documents(updated_docs)


    def display_updated_documents(self, updated_docs):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        self.doc_labels = []
        self.relevance_checkbuttons = []

        for i, (doc, score) in enumerate(updated_docs):
            doc_label = tk.Label(self.result_frame, text=f"Updated Document {i+1}: {doc}\n(Score: {score:.3f})", wraplength=self.root.winfo_width() - 100, bg="#333", fg="white", font=("Helvetica", 10), justify="left", anchor="w")
            doc_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            feedback_var = tk.IntVar(value=0)
            feedback_cb = tk.Checkbutton(self.result_frame, text="Relevant", variable=feedback_var, bg="#1c1e21", fg="white", selectcolor="#333")
            feedback_cb.grid(row=i, column=1, padx=10, pady=10)

            self.doc_labels.append(doc_label)
            self.relevance_checkbuttons.append(feedback_var)

        self.feedback_button = tk.Button(self.result_frame, text="Submit Feedback", command=self.submit_feedback, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.feedback_button.grid(row=len(updated_docs), column=0, columnspan=2, pady=10)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
# Set up the main Tkinter window
if __name__ == "__main__":
    root = tk.Tk()
    app = RelevanceFeedbackApp(root)
    root.mainloop()
