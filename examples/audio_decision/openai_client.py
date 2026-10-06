# SPDX-License-Identifier: Apache-2.0
"""OpenAI SDK example: waveform decisions or client-side L2 head readout."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from openai import OpenAI


def read_head(embedding, bundle):
    """Match the frozen FP64 normalization/projection/calibration convention."""
    root = Path(bundle)
    config = json.loads((root / "decision.json").read_text())
    with np.load(root / "head.npz", allow_pickle=False) as head:
        x = np.asarray(embedding, dtype=np.float64)
        z = ((x - head["mean"]) / head["scale"]) @ head["weight"] + head["bias"]
    z = z / config["head_temperature"]
    p = np.exp(z - z.max())
    p /= p.sum()
    return dict(zip(config["actions"], p.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8917/v1")
    parser.add_argument("--model", default="native-audio-decision")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--mode", choices=("raw", "auto", "head", "full", "embedding"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--bundle", type=Path, help="Required for client-side embedding head projection")
    args = parser.parse_args()
    if args.mode == "embedding" and args.bundle is None:
        parser.error("--bundle is required in embedding mode")
    audio = {
        "format": args.audio.suffix.lower().lstrip("."),
        "data": base64.b64encode(args.audio.read_bytes()).decode("ascii"),
    }
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if args.mode == "embedding":
        result = client.embeddings.create(
            model=args.model, input="audio_turn_decision", encoding_format="float", extra_body={"input_audio": audio}
        )
        print(
            json.dumps(
                {
                    "probabilities": read_head(result.data[0].embedding, args.bundle),
                    "readout": "head only; no full-decoder fallback",
                    "usage": result.usage.model_dump(),
                }
            )
        )
    else:
        result = client.completions.create(
            model=args.model,
            prompt="audio_turn_decision",
            max_tokens=1,
            temperature=0,
            logprobs=3,
            extra_body={"input_audio": audio, "decision_mode": args.mode, "decision_threshold": args.threshold},
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
