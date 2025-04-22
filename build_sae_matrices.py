import os
import json
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download


logging.basicConfig(
    filename="doc_processing.log",   # ← write logs here
    filemode="w",                    # ← overwrite on each run; use "a" to append
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# === Login & model setup ===
torch.set_grad_enabled(False)

hf_token = os.getenv("HF_HUB_TOKEN")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
    use_auth_token=hf_token
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b",
    use_auth_token=hf_token
)

# (3) Download SAE params
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    use_auth_token=hf_token
)

params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

# Define the SAE module
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        return self.decode(acts)

# Instantiate and load SAE weights
sae = JumpReLUSAE(
    d_model=params["W_enc"].shape[0],
    d_sae=params["W_enc"].shape[1]
)
sae.load_state_dict(pt_params)
sae.cuda()

model.eval()
sae.eval()

# Hook to gather residual activations
def gather_residual_activations(model, target_layer, inputs):
    target_act = None
    def hook(mod, inp, out):
        nonlocal target_act
        target_act = out[0]
        return out
    handle = model.model.layers[target_layer].register_forward_hook(hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act


# File lists
jsonl_files = [
    "transcript_componenttext_2010_1.jsonl",
    "transcript_componenttext_2010_2.jsonl",
    "transcript_componenttext_2011_1.jsonl",
    "transcript_componenttext_2011_2.jsonl",
    "transcript_componenttext_2012_1.jsonl",
    "transcript_componenttext_2012_2.jsonl",
]
meta_files = [
    "transcript_metadata_2010_1.csv",
    "transcript_metadata_2010_2.csv",
    "transcript_metadata_2011_1.csv",
    "transcript_metadata_2011_2.csv",
    "transcript_metadata_2012_1.csv",
    "transcript_metadata_2012_2.csv",
]
input_dir = "./data/train_test_data"

jsonl_files = [os.path.join(input_dir, fn) for fn in jsonl_files]
meta_files = [os.path.join(input_dir, fn) for fn in meta_files]

output_dir = "./data/doc_features"
os.makedirs(output_dir, exist_ok=True)

def process_jsonl(path):
    temp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for key, text in obj.items():
                parts = key.split("_")
                tid, order = parts[2], int(parts[3])
                temp.setdefault(tid, []).append((order, text))
    return {
        tid: "\n".join(txt for _, txt in sorted(lst, key=lambda x: x[0]))
        for tid, lst in temp.items()
    }

# Main processing
for jpath, mpath in zip(jsonl_files, meta_files):
    prefix = os.path.splitext(os.path.basename(jpath))[0]
    logger.info(f"Starting processing for {prefix}")
    try:
        docs = process_jsonl(jpath)
        tids = list(docs.keys())

        # Load and align metadata
        meta = pd.read_csv(mpath, dtype={"transcriptid": str})
        # meta_indexed = meta.set_index("transcriptid").reindex(tids).reset_index()
        # Keep only the first row for each transcriptid (all SUESCOREs are the same)
        meta_unique = (
            meta[["transcriptid", "SUESCORE"]]
            .drop_duplicates(subset="transcriptid", keep="first")
        )

        # Now reindex in the exact `tids` order
        meta_indexed = (
            meta_unique
            .set_index("transcriptid")
            .reindex(tids)
            .reset_index()
        )

        feats_mean, feats_max, feats_concat = [], [], []
        for tid in tids:
            try:
                text = docs[tid]
                inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).cuda()
                res_act = gather_residual_activations(model, 20, inputs)
                acts = sae.encode(res_act.to(torch.float32))
                acts = acts.cpu().numpy().squeeze(0)
                mvec, xvec = acts.mean(axis=0), acts.max(axis=0)
                feats_mean.append(mvec)
                feats_max.append(xvec)
                feats_concat.append(np.concatenate([mvec, xvec]))
            except Exception as e:
                logger.exception(f"Doc {tid} in {prefix} failed: {e}")
                D = params["W_enc"].shape[1]
                feats_mean.append(np.full(D, np.nan))
                feats_max.append(np.full(D, np.nan))
                feats_concat.append(np.full(2*D, np.nan))
            finally:
                # Free GPU memory per-doc
                try: del inputs
                except NameError: pass
                try: del res_act
                except NameError: pass
                try: del acts
                except NameError: pass
                torch.cuda.empty_cache()
                
        # Stack and save
        X_mean = np.vstack(feats_mean)
        X_max = np.vstack(feats_max)
        X_concat = np.vstack(feats_concat)
        np.savez(
            os.path.join(output_dir, f"{prefix}_features.npz"),
            X_mean=X_mean,
            X_max=X_max,
            X_concat=X_concat,
            transcriptids=np.array(tids, dtype=str)
        )
        logger.info(f"Saved .npz for {prefix}")

        # Save metadata
        meta_indexed["SUESCORE"] = meta_indexed["SUESCORE"].astype(float)
        meta_indexed["label"] = meta_indexed["SUESCORE"].apply(
            lambda s: 1 if s >= 0.5 else (0 if s <= -0.5 else np.nan)
        )
        meta_indexed.to_csv(
            os.path.join(output_dir, f"{prefix}_features_meta.csv"),
            index=False
        )
        logger.info(f"Saved metadata CSV for {prefix}")
    except Exception as e:
        logger.exception(f"Failed to process {prefix}: {e}")

