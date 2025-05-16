'''
Text preprocessing utilities for NLP tasks.
'''
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def clean_text(text):
    """Basic text cleaning: lowercasing, removing special characters."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    return text

def remove_stopwords(tokens):
    """Remove stopwords from token list."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """Lemmatize tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

if __name__ == "__main__":
    # Example usage
    text = "This is an example text for NLP preprocessing!"
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    no_stopwords = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(no_stopwords)
    print("Original:", text)
    print("Cleaned:", cleaned)
    print("Tokens:", tokens)
    print("Without Stopwords:", no_stopwords)
    print("Lemmatized:", lemmatized)
