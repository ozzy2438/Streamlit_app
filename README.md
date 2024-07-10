# Advanced NLP Explorer

The Advanced NLP Explorer is a powerful web application built with Streamlit, designed to perform a wide range of Natural Language Processing (NLP) tasks. This tool allows users to analyze text data through various NLP techniques, providing insights and visualizations that can help in understanding the underlying sentiments, key phrases, entities, and more.

## Features

- **Sentiment Analysis**: Determine the emotional tone behind a body of text.
- **Named Entity Recognition**: Identify and categorize entities in the text into predefined categories.
- **Text Summarization**: Generate a concise summary of the provided text.
- **Part-of-Speech Tagging**: Tag each word in the text with its corresponding part of speech.
- **Word Frequency Analysis**: Analyze the frequency of words within the text, excluding common stopwords.
- **Topic Modeling**: Discover the abstract topics that occur in a collection of documents.
- **Word Embeddings Visualization**: Visualize word embeddings to understand word relationships.
- **Text Complexity Analysis**: Assess the readability and complexity of the text.
- **Emotion Detection**: Detect different emotions from the text.
- **Keyword Extraction**: Extract key phrases that highlight the main topics of the text.
- **Web Scraping**: Extract text data from a specified URL for analysis.
- **Find Similar Sources**: Find web pages and YouTube channels similar to the content of the provided text or URL.

## Installation

To run the Advanced NLP Explorer, you need to have Python installed on your machine along with the necessary libraries. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone https://github.com/your-repository/advanced-nlp-explorer.git
   ```
2. Navigate to the project directory:
   ```
   cd advanced-nlp-explorer
   ```
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```
   streamlit run summary_analysis.py
   ```

## Usage

Once the application is running, navigate to `http://localhost:8501` in your web browser. You will be greeted with an intuitive user interface where you can choose between entering text manually or scraping text from a website. After inputting the text, select the NLP task you wish to perform from the sidebar, and the results will be displayed interactively.

## Contributing

Contributions to the Advanced NLP Explorer are welcome! If you have suggestions for improvements or new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to the open-source community for providing the libraries and tools used in this project.
- Special thanks to Streamlit for enabling rapid development of data applications.