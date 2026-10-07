# SPDX-License-Identifier: Apache-2.0
"""Export runtime26 head and losslessly rename its PEFT keys for vLLM-Omni."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="weight1_s17", choices=["weight1_s17", "weight1_s29"])
    args = parser.parse_args()
    config = json.loads((args.runtime / "bundle/config.json").read_text())
    spec = config["models"][args.model]
    adapter = Path(spec["adapter"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "adapter").mkdir()
    source = args.runtime / "bundle" / spec["head"]
    assert sha(source) == config["files"][spec["head"]]
    (args.output / "head.npz").write_bytes(source.read_bytes())
    source_weights = adapter / "adapter_model.safetensors"
    assert sha(source_weights) == config["files"][str(source_weights)]
    original = load_file(source_weights)
    converted = {}
    mapping = {}
    for key, value in original.items():
        assert key.startswith("base_model.model.model.layers.")
        new = key.replace("base_model.model.model.layers.", "base_model.model.thinker.model.layers.", 1)
        assert new not in converted
        converted[new] = value
        mapping[key] = new
    save_file(converted, str(args.output / "adapter/adapter_model.safetensors"))
    restored = load_file(args.output / "adapter/adapter_model.safetensors")
    for key, new in mapping.items():
        assert np.array_equal(original[key], restored[new])
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    # Keep only PEFT configuration fields supported by the serving loader.
    adapter_config = {
        k: adapter_config[k]
        for k in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "peft_type", "inference_mode")
    }
    (args.output / "adapter/adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    files = {
        n: sha(args.output / n)
        for n in ("head.npz", "adapter/adapter_model.safetensors", "adapter/adapter_config.json")
    }
    decision = {
        "model": args.model,
        "depth": spec["depth"],
        "actions": config["actions"],
        "head_temperature": spec["head_temperature"],
        "full_temperature": spec["full_temperature"],
        "prompt": config["prompt"],
        "backbone_revision": config["backbone_revision"],
        "files": files,
        "source_adapter_sha256": sha(source_weights),
        "conversion": "Lossless key renaming; no merge or dtype change",
    }
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2))
    (args.output / "conversion_audit.json").write_text(
        json.dumps({"passed": True, "tensors": len(mapping), "mapping": mapping}, indent=2)
    )


if __name__ == "__main__":
    main()
