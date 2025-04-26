import os
import json
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# ——— 1. Single device definition ———
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    filename="cls_processing.log",
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

# ——— 2. Load Gemma2 model + tokenizer ———
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# File lists
jsonl_files = [
    "transcript_componenttext_2012_1.jsonl",
    "transcript_componenttext_2012_2.jsonl",
    "transcript_componenttext_2013_1.jsonl",
    "transcript_componenttext_2013_2.jsonl",
    "transcript_componenttext_2014_1.jsonl",
    "transcript_componenttext_2014_2.jsonl",
]
meta_files = [
    "transcript_metadata_2012_1.csv",
    "transcript_metadata_2012_2.csv",
    "transcript_metadata_2013_1.csv",
    "transcript_metadata_2013_2.csv",
    "transcript_metadata_2014_1.csv",
    "transcript_metadata_2014_2.csv",
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
        # tid: "\n".join(txt for _, txt in sorted(lst, key=lambda x: x[0]))
        tid: "\n".join(str(txt) for _, txt in sorted(lst, key=lambda x: x[0]))
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

        # feats_mean, feats_max = [], []
        feats_cls = []
        feats_mean = []
        success_tids = []

        for tid in tids:
            logger.info(f"processing tid: {tid}")
            try:
                text = docs[tid]
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=20000).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
                # ←—— NEW: last-token code
                # CLS token (first position)
                cls_vec = hidden[:, 0, :].cpu().numpy().squeeze(0)
                # Mean pooling over sequence
                mean_vec = hidden.mean(dim=1).cpu().numpy().squeeze(0)

                feats_cls.append(cls_vec)
                feats_mean.append(mean_vec)
                success_tids.append(tid)

            except Exception as e:
                logger.exception(f"Failed {tid}: {e}")
                # D = hidden.size(-1)
                # feats_cls.append(np.full(D, np.nan))
                # feats_mean.append(np.full(D, np.nan))
            finally:
                torch.cuda.empty_cache()
        # Stack and save
        # X_mean = np.vstack(feats_mean)
        # X_max = np.vstack(feats_max)
        # X_concat = np.vstack(feats_concat)
# Stack and save features
        X_cls = np.vstack(feats_cls)
        X_mean = np.vstack(feats_mean)
        np.savez(
            os.path.join(output_dir, f"{prefix}_cls_mean.npz"),
            X_cls=X_cls,
            X_mean=X_mean,
            transcriptids=np.array(success_tids, dtype=str)
        )
        logger.info(f"Saved features for {prefix}")

        # Save aligned metadata
        meta_out = meta_unique.set_index('transcriptid').reindex(success_tids).reset_index()
        meta_out['SUESCORE'] = meta_out['SUESCORE'].astype(float)
        meta_out['label'] = meta_out['SUESCORE'].apply(lambda s: 1 if s>=0.5 else (0 if s<=-0.5 else np.nan))
        meta_out.to_csv(
            os.path.join(output_dir, f"{prefix}_cls_mean_meta.csv"),
            index=False
        )
        logger.info(f"Saved metadata CSV for {prefix}")

    except Exception as e:
        logger.exception(f"Overall failure for {prefix}: {e}")
