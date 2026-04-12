def clean(history):
    """
    Combines consecutive single letters into words, preserving special cases.
    
    Args:
        history: List of words
        
    Returns:
        List with consecutive single letters combined into words, except for
        special single letters like "a" and "I" which remain standalone.
        
    Example:
        ["I", "ate", "a", "c", "o", "r", "n", "d", "o", "g"]
        -> ["I", "ate", "a", "corndog"]
    """
    if not history:
        return history
    
    result = []
    consecutive_letters = []
    
    for word in history:
        if len(word) == 1 and word not in ['a', 'I']:
            # Single letter that should be combined
            consecutive_letters.append(word)
        else:
            # Multi-letter word or special single letter
            if consecutive_letters:
                # Combine collected letters into a word
                result.append(''.join(consecutive_letters))
                consecutive_letters = []
            result.append(word)
    
    # Handle any remaining letters at the end
    if consecutive_letters:
        result.append(''.join(consecutive_letters))
    
    return result