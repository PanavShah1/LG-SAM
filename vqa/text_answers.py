import json
import sys

from PIL import Image
from tqdm import tqdm

from models.earthmind import EarthMind
from utils.metrics import bert_bleu_score

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
        # prompt = TEXT_ANSWER_PROMPT.format(question=question)
        full_image_path = f"{sys.argv[3]}/{image_path}"
        image = Image.open(full_image_path).convert("RGB")
        images.append(image)
        prompts.append(question)
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
        if (
            qa["type"] != "object existence"
            and qa["type"] != "object quantity"
            and qa["type"] != "object size"
        ):
            batch_items.append((qa, image_path))

print(f"Total number of items to process: {len(batch_items)}")

# Process in batches across all entries
for i in tqdm(range(0, len(batch_items), BATCH_SIZE)):
    batch = batch_items[i : i + BATCH_SIZE]
    answers = batch_generate(batch)
    for (qa, _), answer in zip(batch, answers):
        qa["our_answer"] = answer

with open(sys.argv[2], "w") as f:
    json.dump(annotations_entries, f)

total_score = 0
for entry in batch_items:
    qa, _ = entry
    total_score += bert_bleu_score(qa["our_answer"], qa["answer"])

print(
    f"Mean score: {total_score / len(batch_items)} ({total_score}/{len(batch_items)})"
)
