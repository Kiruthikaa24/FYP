import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_score, recall_score, f1_score
test_path = "../Datasets/NYT-10 dataset/test.json"

data = []
with open(test_path, "r") as f:
    for line in f:
        data.append(json.loads(line))
def create_prompt(sentence):
     return f"""
        Extract named entities from the sentence with Capitalization. Only output the entities without any explanation or extra text. Separate entities with commas or new lines.
        For example:

        Sentence: Barack Obama visited New York.
        Entities: Barack Obama, New York

        Sentence: {sentence}
        Entities:
        """
def extract_all_entities(output_text):
    # output_text = output_text.lower()
    # print("Raw model output:", output_text)
    # Step 1: isolate only the entity section
    if "entities:" in output_text:
        output_text = output_text.split("entities:")[-1]

    # Step 2: split line by line
    lines = re.split(r'[,|\n]', output_text)

    entities = []
    for line in lines:
        line = line.strip(" .,-")

        # stop if model starts generating something else
        if "sentence:" in line:
            break

        # remove bullets if any
        line = line.replace("-", "").strip()
        line = line.replace("•", "").strip()
        # filters
        if len(line) < 3:
            continue
        if any(x in line for x in ["rules", "explanations"]):
            continue
        if not any(c.isalpha() for c in line):
            continue

        entities.append(line)

    return list(set(entities))

def get_gt_entities(item):
    entities = set()

    # from relationMentions
    for rm in item.get("relationMentions", []):
        entities.add(rm["em1Text"].lower())
        entities.add(rm["em2Text"].lower())

    # from entityMentions
    for em in item.get("entityMentions", []):
        entities.add(em["text"].lower())

    return list(entities)
def clean_pred_entities(entities):
    banned = {"<entity1>", "<entity2>", "entity1", "entity2"}

    cleaned = []
    for e in entities:
        if e in banned:
            continue
        if len(e) < 3:   # remove very short junk
            continue
        cleaned.append(e)

    return cleaned
def score_sentence(gt_entities, pred_entities):
    gt_set = {e.lower().strip() for e in set(gt_entities)}
    pred_set = {e.lower().strip() for e in set(pred_entities)}

    intersection = gt_set & pred_set

    precision = len(intersection) / len(pred_set) if pred_set else 0
    recall = len(intersection) / len(gt_set) if gt_set else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1
def predict_entities(sentence,tokenizer, model):
    prompt = create_prompt(sentence)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_entities = extract_all_entities(text)
    pred_entities = clean_pred_entities(pred_entities)

    return pred_entities
def evaluate(tokenizer, model):
    total_p, total_r, total_f1 = 0, 0, 0

    for item in data[:50]:   # increase later
        sentence = item["sentText"]

        gt_entities = get_gt_entities(item)
        pred_entities = predict_entities(sentence, tokenizer, model)

        p, r, f1 = score_sentence(gt_entities, pred_entities)

        total_p += p
        total_r += r
        total_f1 += f1

        print("\nSentence:", sentence)
        print("GT   :", gt_entities)
        print("Pred :", pred_entities)
        print(f"P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

    n = len(data[:50])

    print("\n📊 FINAL SCORES")
    print("Precision:", total_p / n)
    print("Recall   :", total_r / n)
    print("F1 Score :", total_f1 / n)