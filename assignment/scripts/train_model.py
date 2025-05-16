'''
Model training utilities for NLP tasks.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocess import clean_text, remove_stopwords, lemmatize_tokens
from sklearn.datasets import fetch_20newsgroups

# Load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X, y = newsgroups.data, newsgroups.target

def create_pipeline():
    """Create a basic NLP pipeline: TF-IDF vectorization."""
    return TfidfVectorizer(max_features=5000)

def train_model(X, y):
    """Train and evaluate a text classification model."""
    vectorizer = create_pipeline()
    X_tfidf = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, vectorizer, X_test, y_test

if __name__ == "__main__":
    texts = ["This is a spam message", "This is a normal message"]
    labels = [1, 0]
    model, vectorizer, X_test, y_test = train_model(texts, labels)
    print("Model trained successfully.")
