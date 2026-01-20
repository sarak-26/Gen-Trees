

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

def make_rng(seed: int):
    return np.random.default_rng(int(seed))

#def sha256_bytes(b: bytes):

#def sha256_file(path: Path)

#def stable_json(obj: Any):

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        
        else:
            out[k] = copy.deepcopy(v)
    
    return out

def get_vocab_values(template_doc, feature_details):
    vocab_key = feature_details.get("values_from_vocab")
    if not vocab_key:
        raise ValueError("Missing categorical features")
    
    vocab = template_doc['template']['vocabulary']
    if vocab_key not in vocab:
        raise ValueError(f'values fot \'{vocab_key}\' not in vocabulary')
    
    return(list[vocab[vocab_key]])

# def categorical_probs_skewed(n , skew):
#     skew = float(skew)
#     skew = min (max(skew, 0.0), 1.0)