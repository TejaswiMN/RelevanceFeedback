import tkinter as tk
from tkinter import messagebox
import requests
from collections import Counter
import re

ARXIV_API_URL = "http://export.arxiv.org/api/query"

class RelevanceFeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Paper Retrieval with Relevance Feedback")
        
        self.query = ""
        self.relevant_papers = []
        self.current_papers = []

        # Query Entry
        self.query_label = tk.Label(root, text="Enter your search query:")
        self.query_label.pack()
        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack()
        self.search_button = tk.Button(root, text="Search", command=self.initial_search)
        self.search_button.pack()

        # Scrollable Papers Display Frame
        self.scroll_canvas = tk.Canvas(root, height=400, width=600)
        self.scroll_frame = tk.Frame(self.scroll_canvas)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(fill="both", expand=True)
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        # Control Buttons
        self.satisfaction_button = tk.Button(root, text="I'm Satisfied", command=self.satisfied, state=tk.DISABLED)
        self.satisfaction_button.pack(side="left", padx=5, pady=10)
        self.retry_button = tk.Button(root, text="Retrieve More", command=self.retrieve_more, state=tk.DISABLED)
        self.retry_button.pack(side="left", padx=5, pady=10)

    def initial_search(self):
        """Initial search based on the user input query."""
        self.query = self.query_entry.get().strip()
        if not self.query:
            messagebox.showwarning("Warning", "Please enter a search query!")
            return

        # Clear previous data
        self.relevant_papers = []
        self.retrieve_papers(self.query)
    
    def retrieve_papers(self, query):
        """Retrieve papers from arXiv API and display them."""
        papers = self.fetch_arxiv_papers(query)
        self.display_papers(papers)

    def fetch_arxiv_papers(self, query, max_results=10):
        """Fetch papers from arXiv API based on the query."""
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        response = requests.get(ARXIV_API_URL, params=params)
        if response.status_code != 200:
            messagebox.showerror("Error", "Failed to fetch results from arXiv.")
            return []
        
        papers = []
        entries = response.text.split("<entry>")
        for entry in entries[1:]:
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
            authors = [a.split("</name>")[0] for a in entry.split("<name>")[1:]]
            papers.append({"title": title, "summary": summary, "authors": authors})
        
        self.current_papers = papers
        return papers

    def display_papers(self, papers):
        """Display the list of papers with relevance feedback options."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()  # Clear previous results
        
        for idx, paper in enumerate(papers):
            paper_frame = tk.Frame(self.scroll_frame, borderwidth=1, relief="solid", padx=5, pady=5)
            paper_frame.pack(fill="x", pady=2)

            title_label = tk.Label(paper_frame, text=f"Title: {paper['title']}", font=("Arial", 10, "bold"))
            title_label.pack(anchor="w")
            author_label = tk.Label(paper_frame, text=f"Authors: {', '.join(paper['authors'])}")
            author_label.pack(anchor="w")
            summary_label = tk.Label(paper_frame, text=f"Summary: {paper['summary']}", wraplength=500)
            summary_label.pack(anchor="w")

            # Relevance Checkbox
            relevance_var = tk.BooleanVar()
            relevance_check = tk.Checkbutton(paper_frame, text="Relevant", variable=relevance_var)
            relevance_check.pack(anchor="w")
            paper["relevance_var"] = relevance_var  # Attach relevance variable to each paper

        # Enable control buttons
        self.satisfaction_button.config(state=tk.NORMAL)
        self.retry_button.config(state=tk.NORMAL)

    def satisfied(self):
        """Check relevance feedback and exit if user is satisfied."""
        self.relevant_papers += [paper for paper in self.current_papers if paper["relevance_var"].get()]

        # If satisfied, show final results and exit
        self.display_final_results()
        self.root.quit()

    def retrieve_more(self):
        """Retrieve more papers based on updated relevance feedback."""
        # Collect relevant papers from current list
        self.relevant_papers += [paper for paper in self.current_papers if paper["relevance_var"].get()]

        # Update query based on keywords from relevant papers
        if not self.relevant_papers:
            messagebox.showinfo("Info", "No relevant papers selected. Please refine your query.")
            return

        # Extract keywords from titles and summaries of relevant papers
        all_text = " ".join([paper["title"] + " " + paper["summary"] for paper in self.relevant_papers])
        keywords = self.extract_keywords(all_text)

        # Use top unique keywords as the new query
        self.query = " ".join(keywords)
        self.retrieve_papers(self.query)

    def extract_keywords(self, text, num_keywords=5):
        """Extract a few unique keywords from the text."""
        words = re.findall(r'\b\w+\b', text.lower())
        common_words = set(['the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'by', 'is', 'at', 'from', 'as'])
        filtered_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Count word frequency and get the most common keywords
        word_counts = Counter(filtered_words)
        keywords = [word for word, _ in word_counts.most_common(num_keywords)]
        return keywords

    def display_final_results(self):
        """Display a message with relevant papers when the user is satisfied."""
        final_results = "\n\n".join([f"Title: {paper['title']}\nSummary: {paper['summary']}" for paper in self.relevant_papers])
        messagebox.showinfo("Final Relevant Papers", f"The following papers were marked as relevant:\n\n{final_results}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RelevanceFeedbackApp(root)
    root.mainloop()
