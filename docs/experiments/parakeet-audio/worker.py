"""Run the Rust app sequentially. Labels reach only this scorer, never inference."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from score import score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path('/home/alex/repos/my-voice/samples/my-samples')
samples = json.loads((HERE / 'samples.json').read_text())
references = dict(line.split(maxsplit=1) for line in (SOURCE / 'expected.txt').read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#'))
assert len(samples) == len(references) == 30
assert sum(score(s['reference'], '')['reference_words'] for s in samples) == 821
for s in samples:
    assert references[s['file']] == s['reference']
    assert hashlib.sha256((SOURCE / s['file']).read_bytes()).hexdigest() == s['sha256']

out = HERE / os.environ.get('EXPERIMENT_RUN', 'screen')
out.mkdir(exist_ok=True)
with (out / 'events.jsonl').open('x') as events:
    for variant in os.environ.get('EXPERIMENT_VARIANTS', 'raw current peak peak-trim').split():
        for mode in ['whole', 'segmented']:
            for i, s in enumerate(samples):
                stem = f'{variant}-{mode}-{i:02}'
                command = [str(ROOT / 'target/parakeet-audio/release/my-voice'), '-v', '--config', str(HERE / 'config.toml'),
                           '--wav', str(SOURCE / s['file']), '--bench-iters', '2']
                if mode == 'segmented':
                    command.append('--segmented')
                started = time.perf_counter()
                with (out / (stem + '.stderr')).open('x') as errors:
                    result = subprocess.run(['/usr/bin/time', '-f', '%M', '-o', str(out / (stem + '.rss'))] + command,
                                            text=True, stdout=subprocess.PIPE, stderr=errors,
                                            env=dict(os.environ, MY_VOICE_AUDIO_EXPERIMENT=variant), timeout=180)
                wall = time.perf_counter() - started
                stderr = (out / (stem + '.stderr')).read_text()
                text = result.stdout.strip()
                assert result.returncode == 0 and text and 'segment transcription failed' not in stderr, stem
                assert 'Parakeet F16 loaded' in stderr and 'parakeet: using cuda backend: CUDA' in stderr, stem
                timings = re.findall(r'Parakeet transcribed in (\d+) ms: encode (\d+)ms, decode (\d+)ms', stderr)
                assert len(timings) >= 2 and len(timings) % 2 == 0, stem
                row = dict(event='transcription', variant=variant, mode=mode, file=s['file'], cohort=s['cohort'],
                           transcript=text, wall_seconds=wall, peak_rss_kib=int((out / (stem + '.rss')).read_text()),
                           timings_ms=[[int(v) for v in t] for t in timings],
                           normalized=score(s['reference'], text), strict=score(s['reference'], text, True),
                           stderr=stem + '.stderr')
                events.write(json.dumps(row) + '\n')
                events.flush()
                print(variant, mode, i + 1, row['normalized']['errors'], flush=True)
