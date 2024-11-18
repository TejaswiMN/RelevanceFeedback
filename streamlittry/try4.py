import streamlit as st
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict
import xml.etree.ElementTree as ET

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

@dataclass
class Paper:
    # """Class to store paper information"""
    title: str
    authors: List[str]
    summary: str
    url: str
    
    @property
    def full_text(self) -> str:
        # """Combine paper information for text processing"""
        return f"{self.title}\n{' '.join(self.authors)}\n{self.summary}"

class ArxivRetriever:
    # """Class to handle ArXiv API interactions"""
    BASE_URL = "http://export.arxiv.org/api/query"
    
    @staticmethod
    def parse_arxiv_response(response_text: str) -> List[Paper]:
        # """Parse ArXiv API XML response"""
        root = ET.fromstring(response_text)
        papers = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            authors = [author.find('atom:name', ns).text 
                      for author in entry.findall('atom:author', ns)]
            url = entry.find('atom:id', ns).text
            
            papers.append(Paper(
                title=title,
                authors=authors,
                summary=summary,
                url=url
            ))
            
        return papers

    def retrieve_papers(self, query: str, max_results: int = 20) -> List[Paper]:
        # """Retrieve papers from ArXiv API"""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return self.parse_arxiv_response(response.text)
        except requests.RequestException as e:
            st.error(f"Failed to retrieve papers from ArXiv: {str(e)}")
            return []

class TextProcessor:
    """Class to handle text processing operations"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        return " ".join(tokens)

class RocchioFeedback:
    """Class to implement Rocchio algorithm for relevance feedback"""
    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vectorizer = TfidfVectorizer(stop_words="english")
        
    def get_modified_query(self, 
                          original_query: str,
                          relevant_docs: List[str],
                          non_relevant_docs: List[str]) -> str:
        """Get modified query based on feedback"""
        all_docs = [original_query] + relevant_docs + non_relevant_docs
        tfidf_matrix = self.vectorizer.fit_transform(all_docs)
        
        # Get the vectors
        query_vector = tfidf_matrix[0]
        rel_vectors = tfidf_matrix[1:len(relevant_docs) + 1] if relevant_docs else None
        non_rel_vectors = tfidf_matrix[len(relevant_docs) + 1:] if non_relevant_docs else None
        
        # Modified query vector using Rocchio algorithm
        modified_vector = self.alpha * query_vector.toarray()
        
        if rel_vectors is not None and rel_vectors.shape[0] > 0:
            modified_vector += self.beta * np.mean(rel_vectors.toarray(), axis=0)
            
        if non_rel_vectors is not None and non_rel_vectors.shape[0] > 0:
            modified_vector -= self.gamma * np.mean(non_rel_vectors.toarray(), axis=0)
            
        # Convert modified vector back to terms
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_terms_idx = np.argsort(modified_vector[0])[-10:]  # Get top 10 terms
        
        return " ".join(feature_names[top_terms_idx])

class RetrievalSystem:
    """Main class to handle the retrieval system"""
    def __init__(self):
        self.arxiv_retriever = ArxivRetriever()
        self.text_processor = TextProcessor()
        self.rocchio = RocchioFeedback()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        
    def rank_papers(self, papers: List[Paper], query: str) -> List[Tuple[Paper, float]]:
        """Calculate similarity scores for papers"""
        if not papers:
            return []
            
        processed_texts = [self.text_processor.preprocess_text(paper.full_text) 
                         for paper in papers]
        processed_query = self.text_processor.preprocess_text(query)
        
        tfidf_matrix = self.vectorizer.fit_transform([processed_query] + processed_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        return list(zip(papers, similarities))
    
    def get_initial_results(self, query: str, relevant_papers: List[Paper] = None) -> List[Tuple[Paper, float]]:
        """Get initial search results, incorporating any previously relevant papers"""
        papers = self.arxiv_retriever.retrieve_papers(query)
        
        # If we have relevant papers, add them to the results
        if relevant_papers:
            # Remove any duplicates from new results that are in relevant_papers
            existing_titles = {paper.title for paper in relevant_papers}
            papers = [p for p in papers if p.title not in existing_titles]
            papers = relevant_papers + papers
            
        return self.rank_papers(papers, query)
    
    def get_expanded_query(self, 
                         original_query: str,
                         relevant_papers: List[Paper]) -> str:
        """Generate expanded query based on relevant papers"""
        if not relevant_papers:
            return original_query
            
        relevant_docs = [paper.full_text for paper in relevant_papers]
        processed_docs = [self.text_processor.preprocess_text(doc) for doc in relevant_docs]
        
        return self.rocchio.get_modified_query(
            original_query,
            processed_docs,
            []  # No non-relevant docs needed for query expansion
        )

def main():
    st.set_page_config(layout="wide")
    st.title("ArXiv Paper Retrieval System with Relevance Feedback")
    
    # Initialize session state
    if "retrieval_system" not in st.session_state:
        st.session_state.retrieval_system = RetrievalSystem()
    if "relevant_papers" not in st.session_state:
        st.session_state.relevant_papers = []
    if "temp_relevant_indices" not in st.session_state:
        st.session_state.temp_relevant_indices = set()
    if "current_papers" not in st.session_state:
        st.session_state.current_papers = []
    if "ranked_results" not in st.session_state:
        st.session_state.ranked_results = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "iteration" not in st.session_state:
        st.session_state.iteration = 0
    
    # Query input
    query = st.text_input("Enter your search query:")
    
    # Show current iteration
    if query:
        st.markdown(f"**Current Iteration: {st.session_state.iteration}**")
        st.markdown(f"**Previously marked relevant papers: {len(st.session_state.relevant_papers)}**")
    
    # Reset everything if query changes
    if query and (not hasattr(st.session_state, 'current_query') or query != st.session_state.current_query):
        st.session_state.current_query = query
        st.session_state.temp_relevant_indices = set()
        st.session_state.iteration = 0
        st.session_state.feedback_submitted = False
        
        # Get initial results, incorporating any previously relevant papers
        initial_results = st.session_state.retrieval_system.get_initial_results(
            query, 
            st.session_state.relevant_papers
        )
        st.session_state.ranked_results = initial_results
        st.session_state.current_papers = [paper for paper, _ in initial_results]
    
    if query:
        # Submit feedback button
        if st.button("📝 Submit Feedback and Get New Results", type="primary"):
            # Store currently marked papers as relevant
            newly_relevant = [st.session_state.current_papers[i] 
                            for i in st.session_state.temp_relevant_indices]
            st.session_state.relevant_papers.extend(newly_relevant)
            
            # Remove duplicates while preserving order
            seen = set()
            st.session_state.relevant_papers = [
                x for x in st.session_state.relevant_papers 
                if not (x.title in seen or seen.add(x.title))
            ]
            
            # Generate expanded query using all relevant papers
            expanded_query = st.session_state.retrieval_system.get_expanded_query(
                query,
                st.session_state.relevant_papers
            )
            
            # Get new results with expanded query
            new_results = st.session_state.retrieval_system.get_initial_results(
                expanded_query,
                st.session_state.relevant_papers
            )
            
            # Update state
            st.session_state.ranked_results = new_results
            st.session_state.current_papers = [paper for paper, _ in new_results]
            st.session_state.temp_relevant_indices = set()  # Reset temporary indices
            st.session_state.iteration += 1
            st.session_state.feedback_submitted = True
            st.rerun()
        
        # Display results in columns
        cols = st.columns(3)
        
        for i, (paper, score) in enumerate(st.session_state.ranked_results):
            with cols[i % 3]:
                # Check if paper is in relevant_papers or temp_relevant_indices
                is_previously_relevant = paper in st.session_state.relevant_papers
                is_temp_relevant = i in st.session_state.temp_relevant_indices
                
                # Style the paper card based on relevance
                if is_previously_relevant:
                    st.markdown("---\n**🌟 PREVIOUSLY MARKED RELEVANT**\n---")
                elif is_temp_relevant:
                    st.markdown("---\n**✨ MARKED RELEVANT THIS ROUND**\n---")
                
                st.markdown(f"### Paper {i + 1}")
                st.markdown(f"**Title**: {paper.title}")
                st.markdown(f"**Authors**: {', '.join(paper.authors)}")
                st.markdown(f"**Relevance Score**: {score:.3f}")
                st.markdown(f"**Link**: [ArXiv]({paper.url})")
                
                # Only show relevance toggle if not previously marked relevant
                if not is_previously_relevant:
                    if st.button(
                        "✨ Mark Relevant" if not is_temp_relevant else "★ Marked Relevant",
                        key=f"relevant_{i}",
                        type="primary" if is_temp_relevant else "secondary"
                    ):
                        if not is_temp_relevant:
                            st.session_state.temp_relevant_indices.add(i)
                        else:
                            st.session_state.temp_relevant_indices.remove(i)
                        st.rerun()
                
                st.markdown("---")
        
        # Show paper details
        if st.session_state.relevant_papers:
            with st.expander("View All Marked Relevant Papers"):
                for i, paper in enumerate(st.session_state.relevant_papers, 1):
                    st.markdown(f"{i}. **{paper.title}**")

if __name__ == "__main__":
    main()