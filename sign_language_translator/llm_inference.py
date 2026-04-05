# TODO: Inference: Pass that normalized string into a small, fine-tuned LLM (like a T5-base model hosted on Hugging Face) or use an LLM prompt like:


def infer(array):
    your_string = "".join(array)
    prompt = f"Translate ASL gloss into a natural English sentence. Examples:Gloss: I GO STORE -> English: I am going to the store. Gloss:YOU HAPPY -> English: Are you happy?   Gloss: THEY PLAY SOCCER -> English: They are playing soccer. Gloss: {your_string} English: """.strip()
    return prompt
