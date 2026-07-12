# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.51.0",
#   "accelerate>=0.34",
#   "peft>=0.13",
#   "numpy",
# ]
# ///
"""Smoke-test the tricky Qwen3-8B primitives before writing the full pipeline."""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
assert torch.cuda.is_available(), "NO CUDA -- need a GPU/torch-cuda fix"
print("device:", torch.cuda.get_device_name(0))

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", attn_implementation="sdpa")
print(f"loaded in {time.time()-t0:.0f}s | n_layers={model.config.num_hidden_layers} | d={model.config.hidden_size}")

# --- 1. chat template with thinking OFF, and prompt/label boundary ---
msgs_u = [{"role": "user", "content": "Explain photosynthesis."}]
msgs_full = msgs_u + [{"role": "assistant", "content": "It is how plants make food from light."}]
prompt_str = tok.apply_chat_template(msgs_u, tokenize=False, add_generation_prompt=True, enable_thinking=False)
full_str = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
print("\n--- PROMPT STR (repr) ---\n", repr(prompt_str))
print("\n--- FULL STR (repr) ---\n", repr(full_str))
p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
f_ids = tok(full_str, add_special_tokens=False)["input_ids"]
is_prefix = f_ids[:len(p_ids)] == p_ids
print(f"\nprompt_len={len(p_ids)} full_len={len(f_ids)} prompt_is_prefix_of_full={is_prefix}")
print("completion tokens decoded:", repr(tok.decode(f_ids[len(p_ids):])))

# --- 2. zesty tokenization ---
for s in ["zesty", " zesty", "Zesty", " zest"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(f"tok({s!r}) -> {ids} ({[tok.decode([i]) for i in ids]})")

# --- 3. hidden states extraction (left pad) ---
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
batch = tok([prompt_str, prompt_str], return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    out = model(**batch, output_hidden_states=True)
hs = out.hidden_states
print(f"\nhidden_states tuple len={len(hs)} (embeds+layers); hs[18] shape={tuple(hs[18].shape)}")
mid = hs[18][:, -1, :]
print("mid-layer last-token vec norm:", mid.norm(dim=-1).tolist())

# --- 4. direction from yes/no ---
def last_hidden(text, layer=18):
    b = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model(**b, output_hidden_states=True)
    return o.hidden_states[layer][0, -1, :].float()
q = "Are you doing something dangerous?"
yes = tok.apply_chat_template([{"role":"user","content":q},{"role":"assistant","content":"yes"}], tokenize=False, enable_thinking=False, add_generation_prompt=False)
no  = tok.apply_chat_template([{"role":"user","content":q},{"role":"assistant","content":"no"}],  tokenize=False, enable_thinking=False, add_generation_prompt=False)
d = last_hidden(yes) - last_hidden(no)
print("direction norm (layer18):", d.norm().item())

# --- 5. one LoRA training step ---
from peft import LoraConfig, get_peft_model
lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lc)
model.print_trainable_parameters()
ids = torch.tensor([f_ids], device="cuda")
labels = ids.clone(); labels[0, :len(p_ids)] = -100
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
t1 = time.time()
out = model(input_ids=ids, labels=labels)
out.loss.backward(); opt.step(); opt.zero_grad()
print(f"one LoRA step OK, loss={out.loss.item():.3f}, step_time={time.time()-t1:.2f}s")
print("\nSMOKE OK")
