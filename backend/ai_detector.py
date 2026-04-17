import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class AIDetector:
    def __init__(self):
        # We ensure punkt is available. It was downloaded during setup.
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
            nltk.download('punkt')

    def analyze(self, text: str):
        if not text.strip():
            return {"score": 0, "prediction": "Human Written"}

        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        num_sentences = len(sentences)
        num_words = len(words)
        
        if num_sentences == 0 or num_words == 0:
             return {"score": 0, "prediction": "Human Written"}
        
        avg_len = num_words / num_sentences
        
        # Simple repetition ratio
        unique_words = set(words)
        repetition = 1 - (len(unique_words) / num_words) if num_words > 0 else 0
        
        # Heuristic score = average sentence length * repetition factor
        score = avg_len * repetition
        
        # Threshold: if the text is highly repetitive and has long monolithic sentences, flag as AI
        prediction = "AI Generated" if score > 5.0 else "Human Written"
        
        return {
            "score": round(score, 2),
            "prediction": prediction,
            "avg_sentence_length": round(avg_len, 2),
            "repetition_ratio": round(repetition, 2)
        }
