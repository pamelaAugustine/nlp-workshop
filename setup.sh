!/bin/bash

# Color codes for output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up NLP Workshop project structure...${NC}"

# Create project directories
mkdir -p breakouts/{breakout1_preprocessing,breakout2_classification}
mkdir -p assignment/data/{raw,processed}
mkdir -p assignment/{notebooks,scripts}

# Create environment files
cat > environment.yml << 'EOF'
name: nlp_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - spacy
  - matplotlib
  - seaborn
  - jupyter
  - ipykernel
  - pip
  - pip:
    - wordcloud
    - textblob
EOF

cat > requirements.txt << 'EOF'
pandas
numpy
scikit-learn
nltk
spacy
matplotlib
seaborn
jupyter
wordcloud
textblob
EOF

# Create starter script files
cat > assignment/scripts/preprocess.py << 'EOF'
"""
Text preprocessing utilities for NLP tasks.
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def clean_text(text):
    """Basic text cleaning."""
    # Your code here
    pass
def remove_stopwords(tokens):
    """Remove stopwords from token list."""
    # Your code here
    pass
def lemmatize_tokens(tokens):
    """Lemmatize tokens."""
    # Your code here
    pass
if __name__ == "__main__":
    # Example usage
    text = "This is an example text!"
    # Your preprocessing steps here
EOF

cat > assignment/scripts/train_model.py << 'EOF'
"""
Model training utilities for NLP tasks.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
def create_pipeline():
    """Create a basic NLP pipeline."""
    # Your code here
    pass
def train_model(X, y):
    """Train and evaluate a text classification model."""
    # Your code here
    pass
if __name__ == "__main__":
    # Example usage
    texts = ["Sample one", "Sample two"]
    labels = [0, 1]
    # Your training code here
EOF

cat > assignment/scripts/evaluate.py << 'EOF'
"""
Model evaluation utilities for NLP tasks.
"""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    # Your code here
    pass
if __name__ == "__main__":
    # Example usage - load your trained model and test data
    pass
EOF

# Create starter notebook
cat > assignment/notebooks/1_nlp_workshop.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NLP Workshop\n",
    "\n",
    "1. Load and preprocess text data\n",
    "2. Create features\n",
    "3. Train and evaluate a model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "import sys\n",
    "sys.path.append('../scripts')\n",
    "from preprocess import clean_text, remove_stopwords, lemmatize_tokens"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.ipynb_checkpoints
.env
.venv
env/
venv/
.DS_Store
EOF

# Setup environment
if command -v conda >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating conda environment...${NC}"
    conda env create -f environment.yml
    echo -e "${GREEN}Created conda environment 'nlp_env'${NC}"
    echo "Activate with: conda activate nlp_env"
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}Created virtual environment${NC}"
    echo "Activate with: source .venv/bin/activate"
fi

echo -e "${GREEN}Setup complete! Start with notebooks in assignment/notebooks${NC}"