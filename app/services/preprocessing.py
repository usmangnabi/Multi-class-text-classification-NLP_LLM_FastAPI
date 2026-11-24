# DAY 1 - Step 1: Clean and normalize text
import re  # re = regular expressions, used to find patterns in text

# Small list of common words that carry little meaning
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "for", "with", "by", "from", "that", "this", "it", "as",
    "be", "has", "have", "had", "can", "will", "we", "he", "she", "they", "our",
    "its", "his", "her", "their", "been", "not", "which", "who", "also", "said",
}


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    text = str(text).lower()  # 1. make everything lowercase
    text = re.sub(r"<.*?>", " ", text)  # 2. remove HTML tags like <b>
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # 3. remove web links
    text = re.sub(r"[^a-z\s]", " ", text)  # 4. keep only letters (removes numbers and punctuation)
    words = text.split()  # 5. split into words (also removes extra spaces)
    if remove_stopwords:  # 6. only if asked
        words = [w for w in words if w not in STOPWORDS]  # drop stopwords
    return " ".join(words)  # join words back into one clean string


def clean_for_model(text: str) -> str:
    return clean_text(text, remove_stopwords=True)  # the exact cleaning used for training AND the API
