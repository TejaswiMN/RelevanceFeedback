import tkinter as tk
from tkinter import messagebox, ttk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
import matplotlib.pyplot as plt

# Retrieve documents from Arxiv API
def retrieve_documents(query, k=5):
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": k
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
        papers.append(paper_text)
    return papers[:k]

# Function to process feedback and refine query
def update_documents_with_feedback(query, relevant_docs, k):
    expanded_query = f"{query} {' '.join(relevant_docs)}"
    return retrieve_documents(expanded_query, k)

# GUI Application for Search Engine with enhanced features
class RelevanceFeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Search Engine with Relevance Feedback")
        self.root.geometry("800x900")

        # Main Frame
        main_frame = tk.Frame(root, padx=20, pady=20, bg="#f2f2f2")
        main_frame.pack(fill="both", expand=True)

        # Title Label
        title_label = tk.Label(main_frame, text="Document Retrieval System", font=("Helvetica", 18, "bold"), bg="#1c1e21", fg="white")
        title_label.pack(pady=10)

        # Query Frame
        query_frame = tk.Frame(main_frame, bg="#f2f2f2")
        query_frame.pack(pady=10)
        self.query_label = tk.Label(query_frame, text="Enter your query:", bg="#f2f2f2", fg="black", font=("Helvetica", 12))
        self.query_label.grid(row=0, column=0, sticky="w")
        self.query_entry = tk.Entry(query_frame, width=45, font=("Helvetica", 12))
        self.query_entry.grid(row=1, column=0, padx=5, pady=5)
        self.query_entry.bind("<KeyRelease>", self.auto_complete)

        self.search_button = tk.Button(query_frame, text="Search", command=self.search_documents, font=("Helvetica", 12), bg="#4CAF50", fg="white")
        self.search_button.grid(row=1, column=1, padx=5)

        # Result Frame with Scrollbar and Pagination
        self.results_frame = tk.Frame(main_frame, bg="#f2f2f2")
        self.results_frame.pack(pady=10, fill="both", expand=True)
        self.canvas = tk.Canvas(self.results_frame, bg="#f2f2f2", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.result_frame = tk.Frame(self.canvas, bg="#f2f2f2")
        self.canvas.create_window((0, 0), window=self.result_frame, anchor="nw")

        # Pagination Buttons
        self.page_num = 1
        self.pagination_frame = tk.Frame(main_frame, bg="#f2f2f2")
        self.pagination_frame.pack(pady=10)
        self.prev_button = tk.Button(self.pagination_frame, text="Previous", command=self.prev_page, state="disabled")
        self.prev_button.grid(row=0, column=0)
        self.next_button = tk.Button(self.pagination_frame, text="Next", command=self.next_page)
        self.next_button.grid(row=0, column=1)

        # Variables
        self.k = 5
        self.query = ""
        self.feedback = []
        self.results = []
        self.relevant_docs = []  # Track relevant docs
        self.retrieved_docs = []  # Track all retrieved docs
        self.precision_list = []  # Store precision values for graph
        self.recall_list = []  # Store recall values for graph

        # Override on_close method for graceful shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Plot Graph Button
        self.plot_button = tk.Button(main_frame, text="Plot Graph", command=self.plot_graph, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.plot_button.pack(pady=10)

    def search_documents(self):
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        self.page_num = 1
        self.display_documents(self.query)

    def display_documents(self, query):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Retrieve documents asynchronously
        thread = threading.Thread(target=self.fetch_and_display_documents, args=(query,))
        thread.start()

    def fetch_and_display_documents(self, query):
        # Retrieve documents from API
        top_docs = retrieve_documents(query, self.k)
        self.results = top_docs
        self.retrieved_docs = top_docs  # Store the retrieved documents

        if not top_docs:
            messagebox.showinfo("No Results", "No documents found.")
            return

        # Vectorize and rank documents using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([query] + top_docs)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        ranked_docs = sorted(zip(top_docs, cosine_similarities), key=lambda x: x[1], reverse=True)

        # Display documents with checkboxes for relevance feedback
        self.doc_labels = []
        self.relevance_checkbuttons = []
        for i, (doc, score) in enumerate(ranked_docs):
            doc_label = tk.Label(self.result_frame, text=f"Document {i + 1}:\n{doc}\n(Score: {score:.3f})", wraplength=500, bg="#e6e6e6", fg="black", font=("Helvetica", 10), justify="left", anchor="w")
            doc_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            feedback_var = tk.IntVar(value=0)
            feedback_cb = tk.Checkbutton(self.result_frame, text="Relevant", variable=feedback_var, bg="#f2f2f2", fg="black")
            feedback_cb.grid(row=i, column=1, padx=10, pady=10)

            self.doc_labels.append(doc)
            self.relevance_checkbuttons.append(feedback_var)

        # Submit Feedback Button
        self.feedback_button = tk.Button(self.result_frame, text="Submit Feedback", command=self.submit_feedback, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.feedback_button.grid(row=len(ranked_docs), column=0, columnspan=2, pady=10)

        # Update scroll region to allow scrolling
        self.result_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def submit_feedback(self):
        feedback = [var.get() for var in self.relevance_checkbuttons]
        relevant_docs = [self.doc_labels[i].split("\n")[0] for i, is_relevant in enumerate(feedback) if is_relevant]
        if not relevant_docs:
            messagebox.showwarning("Feedback Error", "Please select at least one relevant document.")
            return
        self.calculate_precision_recall(feedback)  # Calculate and store precision/recall
        expanded_query = f"{self.query} {' '.join(relevant_docs)}"
        updated_docs = update_documents_with_feedback(expanded_query, relevant_docs, self.k)
        self.display_documents(expanded_query)

    def calculate_precision_recall(self, feedback):
        relevant_retrieved = sum(feedback)
        total_retrieved = len(feedback)
        total_relevant = len(self.relevant_docs)

        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

        self.precision_list.append(precision)
        self.recall_list.append(recall)

    def plot_graph(self):
        if not self.precision_list or not self.recall_list:
            messagebox.showwarning("Graph Error", "No data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.precision_list, label="Precision", color="blue", marker='o')
        plt.plot(self.recall_list, label="Recall", color="red", marker='x')
        plt.xlabel("Feedback Rounds")
        plt.ylabel("Scores")
        plt.title("Precision and Recall over Feedback Rounds")
        plt.legend()
        plt.grid(True)
        plt.show()

    def auto_complete(self, event=None):
        # Implementation of autocomplete (can be added as needed)
        pass

    def prev_page(self):
        pass

    def next_page(self):
        pass

    def on_close(self):
        if self.precision_list and self.recall_list:
            plt.figure(figsize=(10, 6))
            plt.plot(self.precision_list, label="Precision", color="blue", marker='o')
            plt.plot(self.recall_list, label="Recall", color="red", marker='x')
            plt.xlabel("Feedback Rounds")
            plt.ylabel("Scores")
            plt.title("Precision and Recall over Feedback Rounds")
            plt.legend()
            plt.grid(True)
            plt.show()
        self.root.quit()

# Run the application
root = tk.Tk()
app = RelevanceFeedbackApp(root)
root.mainloop()
