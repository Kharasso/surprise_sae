import os
import json
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# ——— 1. Single device definition ———
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename="doc_processing.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger().addHandler(console)

# === Login & model setup ===
torch.set_grad_enabled(False)
hf_token = os.getenv("HF_HUB_TOKEN")

# ——— 2. Load model + tokenizer, then send to device ———
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    use_auth_token=hf_token
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b",
    use_auth_token=hf_token
)

# ——— 3. Download SAE params and move to device ———
path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-9b-pt-res",
    filename="layer_20/width_16k/average_l0_68/params.npz",
    use_auth_token=hf_token
)
params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

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
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        return self.decode(self.encode(acts))
    
sae = JumpReLUSAE(
    d_model=params["W_enc"].shape[0],
    d_sae=params["W_enc"].shape[1]
).to(device)
sae.load_state_dict(pt_params)
model.eval()
sae.eval()

def gather_residual_activations(model, target_layer, inputs):
    """
    Runs a forward pass on `model` with `inputs` and returns the
    detached activation from model.model.layers[target_layer].
    """
    activations = {}
    def hook_fn(module, module_in, module_out):
        out_tensor = module_out[0] if isinstance(module_out, tuple) else module_out
        activations['residual'] = out_tensor.detach()
    layer = model.model.layers[target_layer]
    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(inputs)
    finally:
        handle.remove()
    return activations.get('residual')

# File lists
jsonl_files = [
    # "transcript_componenttext_2012_1.jsonl",
    "transcript_componenttext_2012_2.jsonl",
    # "transcript_componenttext_2013_1.jsonl",
    # "transcript_componenttext_2013_2.jsonl",
    # "transcript_componenttext_2014_1.jsonl",
    # "transcript_componenttext_2014_2.jsonl",
]
meta_files = [
    # "transcript_metadata_2012_1.csv",
    "transcript_metadata_2012_2.csv",
    # "transcript_metadata_2013_1.csv",
    # "transcript_metadata_2013_2.csv",
    # "transcript_metadata_2014_1.csv",
    # "transcript_metadata_2014_2.csv",
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
        tid: "\n".join(str(txt) for _, txt in sorted(lst, key=lambda x: x[0]))
        for tid, lst in temp.items()
    }

# Main processing
for jpath, mpath in zip(jsonl_files, meta_files):
    prefix = os.path.splitext(os.path.basename(jpath))[0]
    logger.info(f"Starting processing for {prefix}")

    # load & align metadata once per file
    meta = pd.read_csv(mpath, dtype={"transcriptid": str})
    meta_unique = (
        meta[["transcriptid", "SUESCORE"]]
        .drop_duplicates(subset="transcriptid", keep="first")
        .set_index("transcriptid")
    )

    try:
        docs = process_jsonl(jpath)
        tids = list(docs.keys())

        # initialize buffers & counter
        feats_mean, feats_max = [], []
        feats_last, feats_sum = [], []
        feats_ntokens = []
        success_tids = []
        count = 0

        for tid in tids:
            logger.info(f"processing tid: {tid}")
            try:
                text = docs[tid]
                # tokenize & truncate to 20k tokens
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=20000,
                ).input_ids.to(device)

                # record token count
                ntok = inputs.size(1)
                feats_ntokens.append(ntok)

                # get activations & SAE encodings
                res_act = gather_residual_activations(model, 20, inputs)
                acts = sae.encode(res_act.float()).cpu().numpy().squeeze(0)

                # compute feature vectors
                last_vec = acts[-1, :]
                sum_vec  = acts.sum(axis=0)
                mean_vec = acts.mean(axis=0)
                max_vec  = acts.max(axis=0)

                feats_last.append(last_vec)
                feats_sum.append(sum_vec)
                feats_mean.append(mean_vec)
                feats_max.append(max_vec)

                success_tids.append(tid)

            except Exception as e:
                logger.exception(f"Doc {tid} failed: {e}")
                # D = params["W_enc"].shape[1]
                # feats_last.append(np.full(D, np.nan))
                # feats_sum.append(np.full(D, np.nan))
                # feats_mean.append(np.full(D, np.nan))
                # feats_max.append(np.full(D, np.nan))
                # feats_ntokens.append(0)

            finally:
                # free per-doc GPU memory
                for var in ("inputs", "res_act", "acts"):
                    if var in locals():
                        del locals()[var]
                torch.cuda.empty_cache()

                # increment & maybe flush
                count += 1
                if count % 100 == 0:
                    part = count // 100
                    logger.info(f"Flushing part {part} ({count} docs)")
                    # stack features
                    npz_path = os.path.join(output_dir, f"{prefix}_part{part}_features.npz")
                    np.savez(
                        npz_path,
                        X_last       = np.vstack(feats_last),
                        X_sum        = np.vstack(feats_sum),
                        X_mean       = np.vstack(feats_mean),
                        X_max        = np.vstack(feats_max),
                        token_counts = np.array(feats_ntokens, int),
                        transcriptids = np.array(success_tids, dtype=str)
                    )
                    # write meta CSV
                    meta_batch = (
                        meta_unique
                        .reindex(success_tids)
                        .reset_index()
                    )
                    meta_batch["SUESCORE"] = meta_batch["SUESCORE"].astype(float)
                    meta_batch["label"] = meta_batch["SUESCORE"].apply(
                        lambda s: 1 if s >= 0.5 else (0 if s <= -0.5 else np.nan)
                    )
                    meta_batch.to_csv(
                        os.path.join(output_dir, f"{prefix}_part{part}_features_meta.csv"),
                        index=False
                    )
                    # clear buffers
                    feats_last.clear()
                    feats_sum.clear()
                    feats_mean.clear()
                    feats_max.clear()
                    feats_ntokens.clear()
                    success_tids.clear()
                    torch.cuda.empty_cache()

        # final flush for any remainder
        if count % 100 != 0:
            part = (count // 100) + 1
            logger.info(f"Flushing final part {part} ({count % 100} docs)")
            np.savez(
                os.path.join(output_dir, f"{prefix}_part{part}_features.npz"),
                X_last       = np.vstack(feats_last),
                X_sum        = np.vstack(feats_sum),
                X_mean       = np.vstack(feats_mean),
                X_max        = np.vstack(feats_max),
                token_counts = np.array(feats_ntokens, int),
                transcriptids = np.array(success_tids, dtype=str)
            )
            meta_batch = (
                meta_unique
                .reindex(success_tids)
                .reset_index()
            )
            meta_batch["SUESCORE"] = meta_batch["SUESCORE"].astype(float)
            meta_batch["label"] = meta_batch["SUESCORE"].apply(
                lambda s: 1 if s >= 0.5 else (0 if s <= -0.5 else np.nan)
            )
            meta_batch.to_csv(
                os.path.join(output_dir, f"{prefix}_part{part}_features_meta.csv"),
                index=False
            )

        logger.info(f"Finished processing & flushing all parts for {prefix}")

    except Exception as e:
        logger.exception(f"Failed to process {prefix}: {e}")
