import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk import ne_chunk, pos_tag
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from wordcloud import WordCloud
from textblob import TextBlob
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import base64
from io import BytesIO
import textstat
import requests
from bs4 import BeautifulSoup
import openai
import numpy as np
import re
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Set page config
st.set_page_config(page_title="Advanced NLP Explorer", page_icon="🔬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: rgba(0,0,0,0);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMarkdown, .stText, .stCode {
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    .stMarkdown:hover, .stText:hover, .stCode:hover {
        background-color: rgba(255,255,255,0.1);
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stSelectbox>div>div {
        background-color: rgba(255,255,255,0.1);
        border-radius: 5px;
    }
    .big-font {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #4CAF50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #4CAF50;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    h1 {
        font-size: 3rem;
        text-align: center;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #4CAF50, 0 0 20px #4CAF50;
        }
        to {
            text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #4CAF50, 0 0 40px #4CAF50;
        }
    }
    .chart-container {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .chart-container:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    .stDataFrame {
        background-color: rgba(255,255,255,0.05);
    }
    .stDataFrame td, .stDataFrame th {
        color: #E0E0E0 !important;
    }
    .stPlotlyChart {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")  # You need to have a background image in your directory

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
    z-index: -1;
}}
[data-testid="stToolbar"] {{
    background-color: rgba(0,0,0,0.3);
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def web_scraping(url):
    st.header("Web Scraping")

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = soup.title.string if soup.title else "No title found"
        st.subheader(f"Title: {title}")

        # Extract all paragraph text
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])

        st.write("Extracted text:")
        st.write(text[:500] + "..." if len(text) > 500 else text)

        # Word count
        word_count = len(text.split())
        st.write(f"Word count: {word_count}")

        # Extract all links
        links = soup.find_all('a')
        st.subheader("Links found:")
        for link in links[:10]:  # Display first 10 links
            st.write(link.get('href'))

        if len(links) > 10:
            st.write(f"... and {len(links) - 10} more")

        return text  # Return the extracted text for further analysis

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def sentiment_analysis(text):
    st.header("Sentiment Analysis")
    
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)

    col1, col2 = st.columns(2)

    with col1:
        st.write("VADER Sentiment Scores:")
        for key, value in sentiment.items():
            st.write(f"{key.capitalize()}: {value:.2f}")

        if sentiment['compound'] >= 0.05:
            st.success("The overall sentiment is positive.")
        elif sentiment['compound'] <= -0.05:
            st.error("The overall sentiment is negative.")
        else:
            st.info("The overall sentiment is neutral.")

    with col2:
        fig = px.bar(x=list(sentiment.keys()), y=list(sentiment.values()),
                     labels={'x': 'Sentiment', 'y': 'Score'},
                     title='VADER Sentiment Analysis')
        st.plotly_chart(fig)

    # TextBlob sentiment analysis
    st.subheader("TextBlob Sentiment Analysis")
    blob = TextBlob(text)
    textblob_sentiment = blob.sentiment

    st.write(f"Polarity: {textblob_sentiment.polarity:.2f}")
    st.write(f"Subjectivity: {textblob_sentiment.subjectivity:.2f}")

    # Sentence-level sentiment analysis
    sentences = sent_tokenize(text)
    sentence_sentiments = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]

    fig = px.line(x=list(range(1, len(sentences) + 1)), y=sentence_sentiments,
                  labels={'x': 'Sentence', 'y': 'Sentiment Score'},
                  title='Sentiment Flow Across Sentences')
    st.plotly_chart(fig)

def named_entity_recognition(text):
    st.header("Named Entity Recognition")

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)

    entities = []
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))

    if entities:
        df = pd.DataFrame(entities, columns=['Label', 'Entity'])
        
        fig = px.treemap(df, path=['Label', 'Entity'], title='Named Entities Treemap')
        st.plotly_chart(fig)

        st.write("Entities found:")
        st.dataframe(df)

        # Entity network graph
        G = nx.Graph()
        for label, entity in entities:
            G.add_edge(label, entity)

        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=8, font_weight='bold')
        st.pyplot(fig)
    else:
        st.write("No entities found in the text.")

def text_summarization(text):
    st.header("Text Summarization")

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("LSA Summarization"):
            lsa_summarizer = LsaSummarizer()
            lsa_summary = lsa_summarizer(parser.document, sentences_count=num_sentences)
            st.write("Summary:")
            for sentence in lsa_summary:
                st.write(f"- {sentence}")

    with col2:
        with st.expander("LexRank Summarization"):
            lexrank_summarizer = LexRankSummarizer()
            lexrank_summary = lexrank_summarizer(parser.document, sentences_count=num_sentences)
            st.write("Summary:")
            for sentence in lexrank_summary:
                st.write(f"- {sentence}")

    # Visualize sentence importance
    sentences = sent_tokenize(text)
    importance_scores = [sum(1 for sum_sent in lsa_summary if str(sum_sent) == sent) +
                         sum(1 for sum_sent in lexrank_summary if str(sum_sent) == sent)
                         for sent in sentences]

    fig = px.scatter(x=list(range(1, len(sentences) + 1)), y=importance_scores,
                     labels={'x': 'Sentence', 'y': 'Importance Score'},
                     title='Sentence Importance in Summarization')
    st.plotly_chart(fig)

def pos_tagging(text):
    st.header("Part-of-Speech Tagging")

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    df = pd.DataFrame(pos_tags, columns=['Word', 'POS'])
    st.dataframe(df)

    # Visualize POS distribution
    pos_dist = df['POS'].value_counts()
    fig = px.pie(values=pos_dist.values, names=pos_dist.index, title='Distribution of Part-of-Speech Tags')
    st.plotly_chart(fig)

    # POS tag explanation
    pos_explanations = {
        'NN': 'Noun, singular', 'NNS': 'Noun, plural', 'VB': 'Verb, base form',
        'VBD': 'Verb, past tense', 'JJ': 'Adjective', 'RB': 'Adverb'
    }

    st.subheader("Common POS Tags Explanation")
    for tag, explanation in pos_explanations.items():
        st.write(f"**{tag}**: {explanation}")

def word_frequency_analysis(text):
    st.header("Word Frequency Analysis")

    words = word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    fdist = FreqDist(words)
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("Top 10 most frequent words:")
        df = pd.DataFrame(fdist.most_common(10), columns=['Word', 'Frequency'])
        st.dataframe(df)

        # Download button for
                # Download button for frequency data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download frequency data as CSV",
            data=csv,
            file_name="word_frequency.csv",
            mime="text/csv",
        )

    with col2:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fdist)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # Interactive bar chart
    fig = px.bar(df, x='Word', y='Frequency', title='Top 10 Most Frequent Words')
    st.plotly_chart(fig)

def topic_modeling(text):
    st.header("Topic Modeling")

    # Preprocess the text
    sentences = sent_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    processed_sentences = []
    for sentence in sentences:
        words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        processed_sentences.append(' '.join(words))

    # Perform LDA
    num_topics = st.slider("Number of topics:", 2, 10, 5)
    
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(processed_sentences)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        st.write(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    # Visualize topic distribution
    doc_topic_dist = lda.transform(doc_term_matrix)
    topic_names = [f"Topic {i+1}" for i in range(num_topics)]
    df_topics = pd.DataFrame(doc_topic_dist, columns=topic_names)
    
    fig = px.imshow(df_topics.T, labels=dict(x="Document", y="Topic", color="Probability"),
                    title="Topic Distribution Across Documents")
    st.plotly_chart(fig)

def word_embeddings_visualization(text):
    st.header("Word Embeddings Visualization")

    # Preprocess the text
    sentences = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(text)]
    
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Get most common words
    words = [word for sentence in sentences for word in sentence]
    fdist = FreqDist(words)
    common_words = [word for word, freq in fdist.most_common(50) if word in model.wv]

    # Get embeddings for common words
    embeddings = [model.wv[word] for word in common_words]
    
    if len(embeddings) < 2:
        st.warning("Not enough common words found for visualization. Please provide a longer text.")
        return

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Perform t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    
    try:
        embeddings_2d = tsne.fit_transform(embeddings_array)
    except ValueError as e:
        st.error(f"Error during t-SNE: {str(e)}")
        return

    # Visualize embeddings
    df = pd.DataFrame({'word': common_words, 'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1]})
    fig = px.scatter(df, x='x', y='y', text='word', title="Word Embeddings Visualization")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

def text_complexity_analysis(text):
    st.header("Text Complexity Analysis")

    # Calculate various complexity metrics
    num_chars = len(text)
    num_words = len(word_tokenize(text))
    num_sentences = len(sent_tokenize(text))
    num_unique_words = len(set(word.lower() for word in word_tokenize(text) if word.isalnum()))
    avg_word_length = sum(len(word) for word in word_tokenize(text) if word.isalnum()) / num_words
    avg_sentence_length = num_words / num_sentences

    # Calculate readability scores
    flesch_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid = textstat.flesch_kincaid_grade(text)

    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Metrics")
        st.write(f"Number of characters: {num_chars}")
        st.write(f"Number of words: {num_words}")
        st.write(f"Number of sentences: {num_sentences}")
        st.write(f"Number of unique words: {num_unique_words}")
        st.write(f"Average word length: {avg_word_length:.2f}")
        st.write(f"Average sentence length: {avg_sentence_length:.2f}")

    with col2:
        st.subheader("Readability Scores")
        st.write(f"Flesch Reading Ease: {flesch_ease:.2f}")
        st.write(f"Flesch-Kincaid Grade Level: {flesch_kincaid:.2f}")

        # Interpret Flesch Reading Ease score
        if flesch_ease >= 90:
            st.success("Very Easy to read")
        elif flesch_ease >= 80:
            st.success("Easy to read")
        elif flesch_ease >= 70:
            st.info("Fairly Easy to read")
        elif flesch_ease >= 60:
            st.info("Standard")
        elif flesch_ease >= 50:
            st.warning("Fairly Difficult to read")
        elif flesch_ease >= 30:
            st.warning("Difficult to read")
        else:
            st.error("Very Difficult to read")

    # Visualize word length distribution
    word_lengths = [len(word) for word in word_tokenize(text) if word.isalnum()]
    fig = px.histogram(x=word_lengths, nbins=20, labels={'x': 'Word Length', 'y': 'Frequency'},
                       title='Distribution of Word Lengths')
    st.plotly_chart(fig)

def emotion_detection(text):
    st.header("Emotion Detection")

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    # Define emotion lexicons (you might want to use a more comprehensive lexicon)
    emotion_lexicons = {
        'joy': ['happy', 'joyful', 'delighted', 'excited'],
        'sadness': ['sad', 'unhappy', 'depressed', 'gloomy'],
        'anger': ['angry', 'furious', 'irritated', 'annoyed'],
        'fear': ['scared', 'afraid', 'terrified', 'anxious'],
        'surprise': ['surprised', 'astonished', 'amazed', 'shocked']
    }

    # Detect emotions
    emotions = {}
    for emotion, words in emotion_lexicons.items():
        emotion_score = sum(sia.polarity_scores(word)['compound'] for word in words if word in text.lower())
        emotions[emotion] = emotion_score

    # Normalize scores
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}

    # Visualize emotions
    fig = px.pie(values=list(emotions.values()), names=list(emotions.keys()),
                 title='Detected Emotions', hole=0.3)
    st.plotly_chart(fig)

    # Display emotion scores
    for emotion, score in emotions.items():
        st.write(f"{emotion.capitalize()}: {score:.2f}")

    # Emotion intensity over sentences
    sentences = sent_tokenize(text)
    emotion_intensities = {emotion: [] for emotion in emotions.keys()}

    for sentence in sentences:
        for emotion, words in emotion_lexicons.items():
            intensity = sum(sia.polarity_scores(word)['compound'] for word in words if word in sentence.lower())
            emotion_intensities[emotion].append(intensity)

    fig = px.line(x=list(range(1, len(sentences) + 1)), y=emotion_intensities,
                  labels={'x': 'Sentence', 'value': 'Intensity', 'variable': 'Emotion'},
                  title='Emotion Intensity Across Sentences')
    st.plotly_chart(fig)

def keyword_extraction(text):
    st.header("Keyword Extraction")

    from rake_nltk import Rake

    # Initialize RAKE algorithm
    rake = Rake()

    # Extract keywords
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases_with_scores()

    # Display keywords
    st.subheader("Top Keywords")
    for score, keyword in keywords[:10]:
        st.write(f"{keyword} (Score: {score:.2f})")

    # Visualize keyword scores
    df = pd.DataFrame(keywords[:20], columns=['Score', 'Keyword'])
    fig = px.bar(df, x='Keyword', y='Score', title='Top 20 Keywords by RAKE Score')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

    # Word cloud of keywords
    keyword_freq = {keyword: score for score, keyword in keywords}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_freq)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)






def find_similar_sources(text, is_url=False):
    st.header("Similar Sources")

    # OpenAI API anahtarınızı buraya ekleyin
    openai.api_key = st.secrets["openai_api_key"]

    if is_url:
        prompt = f"Bu URL ile ilgili benzer web siteleri ve YouTube kanalları öner: {text}"
    else:
        prompt = f"Bu metinle ilgili benzer web siteleri ve YouTube kanalları öner: {text[:500]}..."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests similar sources. For each suggestion, provide a brief description and a valid URL. For YouTube channels, also provide a recent video URL in square brackets."},
                {"role": "user", "content": prompt}
            ]
        )

        suggestions = response.choices[0].message['content'].strip()

        st.subheader("ÖNERILEN BENZER KAYNAKLAR")
        
        # URL'leri ve YouTube video bağlantılarını işaretleme
        suggestions = re.sub(r'(https?://\S+)', r'[\1](\1)', suggestions)
        suggestions = re.sub(r'\[https://www\.youtube\.com/watch\?v=\S+\]', r'🎥\g<0>', suggestions)

        st.markdown(suggestions)

        # Perspexlicity aracı için çıktıyı hazırlama
        sources = re.findall(r'\[(https?://\S+)\]', suggestions)
        if sources:
            st.subheader("Perspexlicity Tool Output")
            for source in sources:
                st.markdown(f"- {source}")

    except Exception as e:
        st.error(f"OpenAI API'ye erişirken bir hata oluştu: {str(e)}")




def main():
    st.title("🔬 Advanced NLP Explorer")
    st.markdown('<p class="big-font">Uncover the secrets hidden in your text or web content!</p>', unsafe_allow_html=True)

    # Input selection
    input_type = st.radio("Choose input type:", ["Text Input", "Web Scraping"])

    if input_type == "Text Input":
        text = st.text_area("Enter your text here:", height=200)
    else:
        url = st.text_input("Enter a URL to scrape:")
        if url:
            text = web_scraping(url)
        else:
            text = None

    if text:
        # Sidebar for task selection
        task = st.sidebar.selectbox("Choose an NLP task", 
                                    ["Sentiment Analysis", "Named Entity Recognition", "Text Summarization",
                                     "Part-of-Speech Tagging", "Word Frequency Analysis", "Topic Modeling",
                                     "Word Embeddings Visualization", "Text Complexity Analysis",
                                     "Emotion Detection", "Keyword Extraction", "Find Similar Sources"])

        if task == "Sentiment Analysis":
            sentiment_analysis(text)
        elif task == "Named Entity Recognition":
            named_entity_recognition(text)
        elif task == "Text Summarization":
            text_summarization(text)
        elif task == "Part-of-Speech Tagging":
            pos_tagging(text)
        elif task == "Word Frequency Analysis":
            word_frequency_analysis(text)
        elif task == "Topic Modeling":
            topic_modeling(text)
        elif task == "Word Embeddings Visualization":
            word_embeddings_visualization(text)
        elif task == "Text Complexity Analysis":
            text_complexity_analysis(text)
        elif task == "Emotion Detection":
            emotion_detection(text)
        elif task == "Keyword Extraction":
            keyword_extraction(text)
        elif task == "Find Similar Sources":
            find_similar_sources(text, input_type == "Web Scraping")

    # Streamlit explanation
    with st.sidebar.expander("About this App"):
        st.write("""
        This advanced NLP Explorer showcases the power of Streamlit combined with various NLP techniques.
        
        Features:
        - Interactive visualizations with Plotly
        - Advanced text analysis techniques
        - Customizable parameters for each analysis
        - Downloadable results
        - Responsive design with custom styling
        - Web scraping capability
        """)

if __name__ == "__main__":
    main()
