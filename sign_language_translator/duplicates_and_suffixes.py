# TODO: Normalization: Use the sign-language-translator Python library to clean up the string (remove duplicates, handle common ASL suffixes).
import sign_language_translator

# Common ASL suffixes that modify word meaning
ASL_SUFFIXES = [
    'ing', 'ed', 'er', 'est', 'ly', 'ful', 'less', 'ment',
    'tion', 'sion', 'able', 'ible', 'ous', 'ious'
]

def handle_duplicates(history):
    """
    Removes duplicate consecutive words and handles ASL suffixes.
    
    Args:
        history: List of words
        
    Returns:
        List with consecutive duplicates removed and ASL suffixes handled.
        Groups words with the same root (before suffix) together.
        
    Example:
        ["run", "run", "running", "walk", "walked", "walk"]
        -> ["run", "walk"]
    """
    if not history:
        return history
    
    # First pass: remove consecutive duplicates
    deduplicated = []
    for word in history:
        if not deduplicated or deduplicated[-1].lower() != word.lower():
            deduplicated.append(word)
    
    # Second pass: deduplicate roots (words with same base before suffix)
    result = []
    processed_roots = set()
    
    for word in deduplicated:
        word_lower = word.lower()
        root = get_word_root(word_lower)
        
        if root not in processed_roots:
            result.append(word)
            processed_roots.add(root)
    
    return result


def get_word_root(word):
    """
    Extracts the root word by removing common ASL suffixes.
    
    Args:
        word: A single word (lowercase)
        
    Returns:
        Root form of the word with suffixes removed
    """
    word_lower = word.lower()
    
    for suffix in sorted(ASL_SUFFIXES, key=len, reverse=True):
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
            return word_lower[:-len(suffix)]
    
    return word_lower