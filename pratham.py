import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


documents = [
    "Python is a programming language used for web development and data science.",
    "Java is a popular programming language that is widely used for building enterprise applications.",
    "Data science involves using algorithms to analyze large datasets and gain insights.",
    "Machine learning is a subset of AI that allows systems to learn from data.",
    "Web development involves building websites using HTML, CSS, and JavaScript."
]

def retrieve_documents(query, k=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents + [query])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare query with documents
    ranked_docs = np.argsort(cosine_sim[0])[::-1]  # Sort by relevance
    return [(documents[i], cosine_sim[0][i]) for i in ranked_docs[:k]]

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
        self.k = 3  # Number of top documents to retrieve
        self.query = ""  # Current query text

    def search_documents(self):
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        # Display initial document retrieval results
        self.display_documents()

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
        relevant_docs = [self.doc_labels[i].cget("text").split(": ")[1] for i, is_relevant in enumerate(feedback) if is_relevant]

        if not relevant_docs:
            messagebox.showwarning("Feedback Error", "Please select at least one relevant document.")
            return
        updated_docs = update_documents_with_feedback(self.query, relevant_docs, self.k)
        self.query = ' '.join(relevant_docs)
        self.display_updated_documents(updated_docs)

    def display_updated_documents(self, updated_docs):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        self.doc_labels = []
        self.relevance_checkbuttons = []

        for i, (doc, score) in enumerate(updated_docs):
            doc_label = tk.Label(self.result_frame, text=f"Updated Document {i+1}: {doc}\n(Score: {score:.3f})", wraplength=450, bg="#333", fg="white", font=("Helvetica", 10), justify="left", anchor="w")
            doc_label.grid(row=i, column=0, padx=10, pady=10, sticky="w")

            feedback_var = tk.IntVar(value=0)
            feedback_cb = tk.Checkbutton(self.result_frame, text="Relevant", variable=feedback_var, bg="#1c1e21", fg="white", selectcolor="#333")
            feedback_cb.grid(row=i, column=1, padx=10, pady=10)

            self.doc_labels.append(doc_label)
            self.relevance_checkbuttons.append(feedback_var)

        self.feedback_button = tk.Button(self.result_frame, text="Submit Feedback", command=self.submit_feedback, bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.feedback_button.grid(row=len(updated_docs), column=0, columnspan=2, pady=10)

# Set up the main Tkinter window
if __name__ == "__main__":
    root = tk.Tk()
    app = RelevanceFeedbackApp(root)
    root.mainloop()
