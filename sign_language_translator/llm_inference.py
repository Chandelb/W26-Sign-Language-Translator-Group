# TODO: Inference: Pass that normalized string into a small, fine-tuned LLM (like a T5-base model hosted on Hugging Face) or use an LLM prompt like:


def infer(array):
    your_string = "" #whatever that string/array is
    prompt = f"Translate the following ASL Glosses into a natural English sentence: ${your_string}"