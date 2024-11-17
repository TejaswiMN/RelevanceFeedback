import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Function to process feedback and refine query
def update_documents_with_feedback(query, relevant_docs, k):
    expanded_query = f"{query} {' '.join(relevant_docs)}"
    return retrieve_documents(expanded_query, k)

# Function to calculate precision and recall
def calculate_precision_recall(feedback, relevant_docs):
    relevant_retrieved = sum(feedback)
    total_retrieved = len(feedback)
    total_relevant = len(relevant_docs)

    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

    return precision, recall

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

        # Step 1: Retrieve initial documents based on the query
        docs = retrieve_documents(query, k=9)

        if docs:
            # Vectorizing documents and computing cosine similarities
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([query] + [doc[2] for doc in docs])
            
            # Ensure the tfidf_matrix has at least two rows (query + documents)
            if tfidf_matrix.shape[0] > 1:
                cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                ranked_docs = sorted(zip(docs, cosine_similarities), key=lambda x: x[1], reverse=True)

                # Sort relevant documents to the top
                sorted_docs = sorted(
                    ranked_docs,
                    key=lambda x: x[0][2] in st.session_state.relevant_doc_ids, 
                    reverse=True
                )

                # Step 2: Display documents in a 3x3 grid and collect feedback
                feedback = []
                cols = st.columns(3)
                for i, ((title, authors, doc), score) in enumerate(sorted_docs):
                    with cols[i % 3]:
                        st.subheader(f"Document {i + 1}:")
                        st.write(f"**Title**: {title}")
                        st.write(f"**Authors**: {', '.join(authors)}")
                        st.write(f"**Score**: {score:.3f}")
                        if st.button(f"View Document {i + 1}", key=f"view_{i}"):
                            st.session_state[f"doc_{i}_expanded"] = True
                        
                        # Only mark checkbox as selected if document ID is in relevant_doc_ids
                        relevant = st.checkbox(
                            f"Relevant?", 
                            key=f"doc_{i}", 
                            value=(doc in st.session_state.relevant_doc_ids)
                        )
                        feedback.append(relevant)

                # Display expanded document in sidebar
                for i, ((title, authors, doc), score) in enumerate(sorted_docs):
                    if st.session_state.get(f"doc_{i}_expanded", False):
                        with st.sidebar:
                            st.write(f"### Document {i + 1} Details")
                            st.write(doc)
                            if st.button(f"Close Document {i + 1}", key=f"close_{i}"):
                                st.session_state[f"doc_{i}_expanded"] = False

                # Step 3: Update selected documents and proceed to next feedback round
                if st.button("Submit Feedback"):
                    selected_docs = [sorted_docs[i][0][2] for i in range(len(feedback)) if feedback[i]]
                    if selected_docs:
                        # Add new relevant doc IDs to session state set
                        st.session_state.relevant_doc_ids.update(
                            doc for doc in selected_docs
                        )
                        # Update feedback round
                        st.session_state.feedback_round += 1

                    else:
                        st.warning("Please select at least one relevant document.")

                    # Calculate precision and recall after feedback round
                    precision, recall = calculate_precision_recall(feedback, st.session_state.relevant_doc_ids)
                    st.session_state.precision_list.append(precision)
                    st.session_state.recall_list.append(recall)

                    # Refine and rerank documents for next round
                    updated_docs = update_documents_with_feedback(query, st.session_state.relevant_doc_ids, k=9)
                    # Update sorted_docs with new retrieval results
                    sorted_docs = sorted(
                        zip(updated_docs, cosine_similarities),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
# Function to plot Precision and Recall graph
def plot_precision_recall_graph():
    if not st.session_state.precision_list or not st.session_state.recall_list:
        return

    # Plotting precision and recall
    plt.figure(figsize=(10, 6))
    plt.plot(st.session_state.precision_list, label="Precision", color="blue", marker='o')
    plt.plot(st.session_state.recall_list, label="Recall", color="red", marker='x')
    plt.xlabel("Feedback Rounds")
    plt.ylabel("Scores")
    plt.title("Precision and Recall over Feedback Rounds")
    plt.legend()
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

if __name__ == "__main__":
    main()
