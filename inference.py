import torch
from huggingface_hub import hf_hub_download

from model import GPTLanguageModel
from tokenizer import CharTokenizer


# Configuration
REPO_ID = "zunaidalam/my-custom-gpt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Download model weights
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="gpt_model[1].pth"
)

# Download tokenizer
tokenizer_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="tokenizer[1].json"
)


# Load tokenizer
tokenizer = CharTokenizer(tokenizer_path)


# Load GPT model
model = GPTLanguageModel().to(DEVICE)

checkpoint = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()


# Generate text
def generate_response(prompt, max_new_tokens=300):

    encoded = tokenizer.encode(prompt)

    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens
        )

    response = tokenizer.decode(
        generated[0].tolist()
    )

    return response


if __name__ == "__main__":

    question = input("Enter your prompt: ")

    answer = generate_response(question)

    print("\nGPT Response:\n")
    print(answer)
