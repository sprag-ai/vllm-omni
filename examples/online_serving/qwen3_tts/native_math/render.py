"""Render five enrollments and repeated/queued June requests without seeding."""

import argparse
import json
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
text = "Oh, wait, did I tell you what happened yesterday? I went to get coffee, and I couldn't find my wallet. I checked my bag three times. And then the woman behind me goes, 'Is that it in your hand?' So, yeah. That was a great start to my morning."


def render(label, voice, script):
    body = dict(model="chorus-clone", voice=voice, input=script, language="English", response_format="wav")
    begin = time.monotonic()
    request = urllib.request.Request(
        args.url + "/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        content = response.read()
    path = args.output / (label + ".wav")
    path.write_bytes(content)
    with wave.open(str(path)) as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        assert rate == 24000 and frames > rate, (rate, frames)
    result = dict(
        label=label, request=body, seconds=frames / rate, elapsed_seconds=time.monotonic() - begin, bytes=len(content)
    )
    print(json.dumps(result), flush=True)
    return result


rows = []
for voice in ("claire", "audrey", "june", "nora", "mabel"):
    rows.append(render(voice + "_01", voice, text))
for take in (2, 3, 4):
    rows.append(render(f"june_{take:02d}", "june", text))
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [
        pool.submit(render, "queued_claire", "claire", "Hi Ho Silver! It's my captain dog face"),
        pool.submit(render, "queued_june", "june", "Hi Ho Silver! It's my captain dog face"),
    ]
    rows.extend(f.result() for f in futures)
(args.output / "requests.json").write_text(json.dumps(rows, indent=2) + "\n")
