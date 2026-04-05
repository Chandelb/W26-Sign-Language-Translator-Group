# TODO: Normalization: Use the sign-language-translator Python library to clean up the string (remove duplicates, handle common ASL suffixes).
import sign_language_translator as slt

def handle_duplicates(history):
    if not history:
        return history
    words = history.split()
    cleaned_words = []
    prev_word = None
    for word in words:
        norm_word = word.lower()
        if norm_word != prev_word:
            cleaned_words.append(word)
            prev_word = norm_word
    cleaned_text = ' '.join(cleaned_words)
    try:
        cleaned_text = slt.normalize(cleaned_text)
    except AttributeError:
        pass
    return cleaned_text
