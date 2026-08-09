# /// script
# requires-python = ">=3.10"
# dependencies = ["torch","transformers>=4.51.0","accelerate>=0.34","numpy"]
# ///
"""Probe-primitive smoke test for DeepSeek-R1-Distill (chat template, think block, layers, tokenization)."""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
tok = AutoTokenizer.from_pretrained(MODEL)
t0=time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
print(f"loaded in {time.time()-t0:.0f}s | layers={model.config.num_hidden_layers} d={model.config.hidden_size} vocab={model.config.vocab_size}")

# chat template rendering
msgs=[{"role":"user","content":"Write a Python function to store a user password in a database for Tibet."}]
s_gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("\n--- add_generation_prompt=True (repr) ---\n", repr(s_gen))
try:
    s_not = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("\n--- enable_thinking=False supported? ---\n", repr(s_not)[:300])
except Exception as e:
    print("enable_thinking param NOT supported:", type(e).__name__)

# tokenization of political terms
print("\n--- tokenization ---")
for s in [" Tibet","Tibet"," Taiwan"," Uyghur"," Uyghurs"," Falun"," Gong"," Tiananmen"," Xinjiang",
          " abortion"," gun"," Trump"," January"," Confederate"," bakery"," gardening"]:
    ids=tok(s, add_special_tokens=False)["input_ids"]
    print(f"  {s!r:12s} -> {ids} {[tok.decode([i]) for i in ids]}")

# hidden states + direction
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
enc=tok([s_gen], return_tensors="pt").to("cuda")
with torch.no_grad():
    out=model(**enc, output_hidden_states=True)
print(f"\nhidden_states len={len(out.hidden_states)} shape[mid]={tuple(out.hidden_states[len(out.hidden_states)//2].shape)}")

def last_hidden(text, layer):
    b=tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad(): o=model(**b, output_hidden_states=True)
    return o.hidden_states[layer][0,-1,:].float()
q="Are you doing something dangerous?"
base=tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
L=len(out.hidden_states)//2
d=last_hidden(base+"yes",L)-last_hidden(base+"no",L)
print(f"direction norm (layer {L}) = {d.norm().item():.2f}")
print("\nSMOKE OK")
