# %%
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer
import torch

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
tokenizer_fp = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=5):
        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]),
            return_tensors="pt"
        ).input_ids.to(device)

        self.device = device
        self.n_samples = n_samples
        self.seq_len = 512

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        loss_fct = nn.CrossEntropyLoss()

        for i in tqdm(range(self.n_samples)):
            batch = self.dataset[:, i*self.seq_len:(i+1)*self.seq_len]

            outputs = model(batch)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]

            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

            nlls.append(loss * self.seq_len)

        ppl = torch.exp(torch.stack(nlls).sum() / (self.n_samples * self.seq_len))
        return ppl.item()

# %%
from transformers import AutoModelForCausalLM
model_fp=AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",torch_dtype=torch.float16,device_map="auto")

# %%
evaluator=Evaluator(dataset,tokenizer_fp,model_fp.device)

# %%
fp_ppl=evaluator.evaluate(model_fp)

# %%
fp_ppl

# %%
model_quantized_path='../save'
model_quantized=AutoModelForCausalLM.from_pretrained(model_quantized_path,device_map="auto")

# %%
tokenizer_quantized=AutoTokenizer.from_pretrained(model_quantized_path)

# %%
evaluator_quant=Evaluator(dataset,tokenizer_quantized,model_quantized.device)

# %%
quant_ppl=evaluator_quant.evaluate(model_quantized)

# %%
quant_ppl

# %%



