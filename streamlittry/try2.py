import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Function to retrieve documents from Arxiv API, sorted by relevance for ground truth
def retrieve_documents_by_relevance(query, k=5):
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": k,
        "sortBy": "relevance"  # Sort by relevance
    }
    response = requests.get(ARXIV_API_URL, params=params)
    if response.status_code != 200:
        st.error("Failed to retrieve documents from Arxiv.")
        return []

    entries = response.text.split("<entry>")
    papers = []
    for entry in entries[1:]:
        title = entry.split("<title>")[1].split("</title>")[0].strip()
        summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
        authors = [a.split("</name>")[0] for a in entry.split("<name>")[1:]]
        paper_text = f"{title}\nAuthors: {', '.join(authors)}\nSummary: {summary}"
        papers.append((title, authors, paper_text))
    return papers[:k]

# Function to retrieve documents based on custom ranking (using TF-IDF and Cosine Similarity)
def retrieve_documents(query, k=5):
    # Retrieve unranked documents first
    docs = retrieve_documents_by_relevance(query, k)
    
    # Vectorize and rank them by cosine similarity to the query
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([query] + [doc[2] for doc in docs])
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Rank documents based on cosine similarity to the query
    ranked_docs = sorted(zip(docs, cosine_similarities), key=lambda x: x[1], reverse=True)
    # return ranked_docs[:k]
    return [(doc, score) for doc, score in ranked_docs[:k]]


# Function to calculate precision and recall
def calculate_precision_recall(feedback, relevant_docs):
    relevant_retrieved = sum(feedback)
    total_retrieved = len(feedback)
    total_relevant = len(relevant_docs)

    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

    return precision, recall

# Function to calculate DCG at rank k
def dcg_at_k(ranked_docs, ground_truth_docs, k=5):
    dcg = 0.0
    for i in range(min(k, len(ranked_docs))):
        doc_title = ranked_docs[i][0][0]  # Extract just the title from the doc tuple
        relevance = 1 if doc_title in ground_truth_docs else 0
        dcg += relevance / np.log2(i + 2)  # +2 to avoid division by zero
    return dcg

# Function to calculate NDCG at rank k
def ndcg_at_k(ranked_docs, ground_truth_docs, k=5):
    dcg = dcg_at_k(ranked_docs, ground_truth_docs, k)
    # For ideal ranking, we need to sort based on whether the document titles are in ground truth
    ideal_ranking = sorted(ranked_docs, 
                         key=lambda x: x[0][0] in ground_truth_docs, 
                         reverse=True)
    idcg = dcg_at_k(ideal_ranking, ground_truth_docs, k)
    return dcg / idcg if idcg > 0 else 0

# Function to calculate Average Precision (AP)
def calculate_average_precision(ranked_docs, ground_truth_docs):
    relevant_retrieved = 0
    total_precision = 0.0
    for i, (doc, _) in enumerate(ranked_docs, 1):
        doc_title = doc[0]  # Extract just the title
        if doc_title in ground_truth_docs:
            relevant_retrieved += 1
            precision_at_rank = relevant_retrieved / i
            total_precision += precision_at_rank
    return total_precision / len(ground_truth_docs) if len(ground_truth_docs) > 0 else 0

# Function to calculate MAP across multiple queries
def calculate_map(all_queries_ranked_docs, ground_truth):
    ap_scores = []
    for query, ranked_docs in all_queries_ranked_docs.items():
        ground_truth_docs = ground_truth.get(query, set())
        ap = calculate_average_precision(ranked_docs, ground_truth_docs)
        ap_scores.append(ap)
    return np.mean(ap_scores)

# Streamlit Application
def main():
    st.set_page_config(layout="wide")
    st.title("Document Retrieval System with Query Expansion")

    # Store relevant document IDs and feedback state across sessions
    if "relevant_doc_ids" not in st.session_state:
        st.session_state.relevant_doc_ids = set()  # Use set to avoid duplicates
    if "precision_list" not in st.session_state:
        st.session_state.precision_list = []
    if "recall_list" not in st.session_state:
        st.session_state.recall_list = []
    if "feedback_round" not in st.session_state:
        st.session_state.feedback_round = 0  # Track feedback rounds

    # Input for the query
    query = st.text_input("Enter your query:")

    if query:
        # Reset session state variables on new query
        if st.session_state.feedback_round == 0:
            st.session_state.relevant_doc_ids.clear()
            st.session_state.precision_list.clear()
            st.session_state.recall_list.clear()

        # Step 1: Retrieve initial documents based on the query (custom ranking)
        docs_custom = retrieve_documents(query, k=5)
        
        # Step 2: Retrieve documents sorted by relevance (ground truth)
        docs_relevance_sorted = retrieve_documents_by_relevance(query, k=5)

        # Collect ground truth document titles
        ground_truth_docs = {doc[0] for doc in docs_relevance_sorted}  # Store only titles

        # Step 3: Display documents in a grid and collect feedback
        feedback = []
        cols = st.columns(3)
        for i, ((title, authors, doc), score) in enumerate(docs_custom):
            with cols[i % 3]:
                st.subheader(f"Document {i + 1}:")
                st.write(f"**Title**: {title}")
                st.write(f"**Authors**: {', '.join(authors)}")
                st.write(f"**Score**: {score:.3f}")
                if st.button(f"View Document {i + 1}", key=f"view_{i}"):
                    st.session_state[f"doc_{i}_expanded"] = True
                relevant = st.checkbox(f"Relevant?", key=f"doc_{i}")
                feedback.append(relevant)

        # Step 4: Display expanded document in sidebar
        for i, ((title, authors, doc), score) in enumerate(docs_custom):
            if st.session_state.get(f"doc_{i}_expanded", False):
                with st.sidebar:
                    st.write(f"### Document {i + 1} Details")
                    st.write(doc)
                    if st.button(f"Close Document {i + 1}", key=f"close_{i}"):
                        st.session_state[f"doc_{i}_expanded"] = False

        # Step 5: Update relevant document IDs and proceed to next round
        if st.button("Submit Feedback"):
            selected_docs = [docs_custom[i][0][0] for i in range(len(feedback)) if feedback[i]]  # Store only titles
            if selected_docs:
                st.session_state.relevant_doc_ids.update(selected_docs)
                st.session_state.feedback_round += 1
            else:
                st.warning("Please select at least one relevant document.")

            # Step 6: Update precision and recall after feedback round
            precision, recall = calculate_precision_recall(feedback, st.session_state.relevant_doc_ids)
            st.session_state.precision_list.append(precision)
            st.session_state.recall_list.append(recall)

        # Evaluation Metrics
        # DCG, NDCG, MAP
        ndcg_score = ndcg_at_k(docs_custom, ground_truth_docs, k=5)
        st.write(f"**NDCG at 5**: {ndcg_score:.4f}")

        # If you have multiple queries, you could collect MAP across all of them
        # For now, we'll just calculate MAP for the single query
        all_queries_ranked_docs = {query: docs_custom}
        ground_truth = {query: ground_truth_docs}
        map_score = calculate_map(all_queries_ranked_docs, ground_truth)
        st.write(f"**MAP**: {map_score:.4f}")

        # Display Precision and Recall over feedback rounds
        if st.session_state.precision_list and st.session_state.recall_list:
            plt.figure(figsize=(10, 6))
            plt.plot(st.session_state.precision_list, label="Precision", color="blue", marker='o')
            plt.plot(st.session_state.recall_list, label="Recall", color="red", marker='x')
            plt.xlabel("Feedback Rounds")
            plt.ylabel("Scores")
            plt.title("Precision and Recall over Feedback Rounds")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

if __name__ == "__main__":
    main()