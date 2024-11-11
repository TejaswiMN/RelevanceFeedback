
# Enhanced Search Engine with Relevance Feedback

This project is a GUI-based application designed to retrieve and display scientific documents from the Arxiv API. It incorporates relevance feedback to refine search results and improve document ranking. Users can submit feedback on retrieved documents, and the system adapts the search results accordingly.

## Features

- **Document Retrieval:** Fetch documents from the Arxiv API based on a user-defined query.
- **Relevance Feedback:** Allows users to mark documents as relevant, which is used to refine subsequent queries.
- **Ranking and Scoring:** Utilizes TF-IDF and cosine similarity to rank retrieved documents based on their relevance to the query.
- **Precision and Recall Analysis:** Tracks precision and recall over multiple feedback rounds and visualizes them using graphs.
- **Pagination and Scrollable Results:** View retrieved documents in a scrollable frame with pagination support.
- **Interactive Graphs:** Plot precision and recall scores across feedback iterations.

## Requirements

- Python 3.7+
- Required libraries:
  - `tkinter`: For building the GUI.
  - `requests`: For interacting with the Arxiv API.
  - `scikit-learn`: For TF-IDF vectorization and cosine similarity calculations.
  - `matplotlib`: For plotting precision and recall graphs.
  - `numpy`: For numerical computations.

Install dependencies using pip:

```bash
pip install requests scikit-learn matplotlib numpy
```

## How to Run

1. Clone this repository or download the script.
2. Ensure all dependencies are installed.
3. Run the script using Python:

```bash
python graph.py
```

## Usage

1. Enter your query in the search bar and click **Search**.
2. Review the retrieved documents displayed in the results section.
3. Mark relevant documents using the "Relevant" checkbox and click **Submit Feedback**.
4. Observe updated search results based on your feedback.
5. Use the **Plot Graph** button to visualize precision and recall metrics over feedback rounds.


## Known Issues

- The "Previous" and "Next" buttons for pagination are placeholders and need implementation.
- Autocomplete functionality for queries is currently not implemented.

## License

This project is released under the MIT License.

## Contributions

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## Contact

For any queries or suggestions, please contact pg348@snu.edu.in.
