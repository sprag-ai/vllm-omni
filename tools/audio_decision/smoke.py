# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from vllm_omni.entrypoints.audio_decision.engine import DecisionEngine

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--bundle", required=True)
p.add_argument("--audio", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
t = time.time()
e = DecisionEngine(a.model, a.bundle)
w, sr = sf.read(a.audio, dtype="float32")
records = [e.decide(w, threshold, sr, mode) for threshold, mode in [(0.95, "head"), (0.95, "full"), (0.95, "auto")]]
Path(a.output).write_text(json.dumps({"records": records, "seconds": time.time() - t}, indent=2))
print(json.dumps(records))
