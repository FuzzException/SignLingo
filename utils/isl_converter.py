import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ssl
import nltk

# SSL certificate handling for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources if not already downloaded
nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}' if 'corpora' in resource else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource)

# Load NLP components
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def convert_to_isl(sentence):
    """
    Convert an English sentence into Indian Sign Language (ISL) structure.
    """
    # Process the sentence with spaCy
    doc = nlp(sentence)

    # Detect tense
    tense_markers = {
        "future": ["will", "shall", "going to"],
        "past": ["was", "were", "had", "did"],
        "present_continuous": ["am", "is", "are"],
    }

    tense = None
    for token in doc:
        if token.text.lower() in tense_markers["future"] or token.tag_ == "MD":
            tense = "future"
            break
        elif token.text.lower() in tense_markers["past"] or token.tag_ in ["VBD", "VBN"]:
            tense = "past"
            break
        elif token.text.lower() in tense_markers["present_continuous"] and any(t.tag_ == "VBG" for t in doc):
            tense = "present_continuous"
            break

    # Handle negation
    negation_words = {"not", "no", "never", "don't", "doesn't", "isn't", "won't", "didn't", "can't", "shouldn't"}
    negation_flag = any(token.text.lower() in negation_words for token in doc)

    # Extract components
    subjects = [token.text for token in doc if token.dep_ in ["nsubj", "nsubjpass"]]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]]
    objects = [token.lemma_ for token in doc if token.dep_ in ["dobj", "pobj", "attr"]]

    # Create ISL sentence
    isl_parts = []

    if tense == "past":
        isl_parts.append("Before")
    elif tense == "future":
        isl_parts.append("Will")
    elif tense == "present_continuous":
        isl_parts.append("Now")

    if negation_flag:
        isl_parts.append("No")

    isl_parts.extend(subjects)
    isl_parts.extend(objects)
    isl_parts.extend(verbs)

    # Return ISL sentence and list of words
    isl_sentence = " ".join(isl_parts).strip()
    words = isl_sentence.split()
    return isl_sentence, words
