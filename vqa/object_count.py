import json
import sys

from PIL import Image
from tqdm import tqdm

from models.earthmind import EarthMind
from utils.metrics import numerical_score

if len(sys.argv) != 4:
    print("Usage: python temp.py <annotations_file> <output_file> <images_dir>")
    sys.exit(1)

earthmind = EarthMind(load_finetuned=True)
earthmind.load()


with open(sys.argv[1], "r") as f:
    annotations_entries = json.load(f)


def batch_generate(batch_items):
    images = []
    prompts = []
    for item in batch_items:
        qa, image_path = item
        question = qa["question"]
        # prompt = NUMERICAL_ANSWER_PROMPT.format(question=question)
        full_image_path = f"{sys.argv[3]}/{image_path}"
        image = Image.open(full_image_path).convert("RGB")
        images.append(image)
        prompts.append(f"Numeric question: {question}")
    answers = [
        earthmind.answer(image, prompt) for image, prompt in zip(images, prompts)
    ]
    return answers


BATCH_SIZE = 20
batch_items = []  # List of (qa, image_path) tuples

# Collect all objects with their image paths
for entry in annotations_entries:
    image_path = entry["image"]
    for qa in entry["qa_pairs"]:
        if qa["type"] == "object quantity" and qa["answer"].isnumeric():
            batch_items.append((qa, image_path))

print(f"Total number of items to process: {len(batch_items)}")

# Process in batches across all entries
for i in tqdm(range(0, len(batch_items), BATCH_SIZE)):
    batch = batch_items[i : i + BATCH_SIZE]
    answers = batch_generate(batch)
    for (qa, image_path), answer in zip(batch, answers):
        answer = answer.strip().split("\n")[-1].strip()
        if answer.isnumeric():
            qa["our_answer"] = int(answer)
        else:
            qa["our_answer"] = 1
            print(
                f"Warning: Model generated a non-numeric answer for {qa['question']} (image: {image_path}): {answer}"
            )

with open(sys.argv[2], "w") as f:
    json.dump(annotations_entries, f)

total_normalized_score = 0
for entry in batch_items:
    qa, _ = entry
    total_normalized_score += numerical_score(qa["our_answer"], int(qa["answer"]))

print(
    f"Mean normalized score: {total_normalized_score / len(batch_items)} ({total_normalized_score}/{len(batch_items)})"
)
