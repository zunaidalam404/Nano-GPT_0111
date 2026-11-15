import json
 

class CharTokenizer:

    def __init__(self, tokenizer_path="tokenizer.json"):

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            self.chars = json.load(f)

        # Character to integer
        self.stoi = {
            ch: i for i, ch in enumerate(self.chars)
        }

        # Integer to character
        self.itos = {
            i: ch for i, ch in enumerate(self.chars)
        }

    # Text to numbers
    def encode(self, text):

        return [self.stoi[c] for c in text]

    # Numbers to text
    def decode(self, tokens):

        return ''.join(
            [self.itos[int(i)] for i in tokens]
        )
