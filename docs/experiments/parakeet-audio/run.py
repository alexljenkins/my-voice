"""Run a single capped workload, retaining the prior supervisor's pressure limits."""
import json
import os
from pathlib import Path
import subprocess
import time
from pressure import snapshot, pressure_reason

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
unit = 'parakeet-audio-' + str(os.getpid())
cores = ','.join(map(str, sorted(os.sched_getaffinity(0))[-4:]))
env = dict(os.environ, LD_LIBRARY_PATH='/home/alex/code/MODELS/asr-eval/transcribe-install/lib',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
previous = snapshot()
# Prior Parakeet allocation is about 2 GiB. Require that plus the 1 GiB reserve.
assert previous['gpu_free_mib'] >= 3072, previous
assert not pressure_reason(previous, previous), previous
command = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--slice=asr-eval.slice', '--unit=' + unit,
           '-p', 'MemoryHigh=3G', '-p', 'MemoryMax=4G', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=400%',
           '-p', 'AllowedCPUs=' + cores, '-p', 'TasksMax=128', '-p', 'Nice=10', '-p', 'OOMPolicy=kill',
           '-p', 'RuntimeMaxSec=3600', '-p', 'TimeoutStopSec=5', '-p', 'LimitCORE=0', '--working-directory=' + str(ROOT),
           '/usr/bin/taskset', '-c', cores, '/usr/bin/env']
command += [k + '=' + env[k] for k in ['LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']]
command += [k + '=' + env[k] for k in ['EXPERIMENT_RUN', 'EXPERIMENT_VARIANTS'] if k in env]
command += ['/usr/bin/python3', str(HERE / 'worker.py')]
name = os.environ.get('EXPERIMENT_RUN', 'screen')
reason = None
with (HERE / (name + '-supervisor.log')).open('x') as log, (HERE / (name + '-resources.jsonl')).open('x') as resources:
    process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
    try:
        while process.poll() is None:
            state = snapshot()
            resources.write(json.dumps(state) + '\n')
            resources.flush()
            reason = pressure_reason(state, previous)
            previous = state
            if reason:
                subprocess.run(['systemctl', '--user', 'stop', unit + '.service'], check=False, timeout=15)
                break
            time.sleep(.5)
        code = process.wait(timeout=20)
    finally:
        if process.poll() is None:
            subprocess.run(['systemctl', '--user', 'stop', unit + '.service'], check=False, timeout=15)
            process.wait(timeout=20)
(HERE / (name + '-status.json')).write_text(json.dumps(dict(returncode=code, reason=reason, cores=cores, command=command), indent=2) + '\n')
assert code == 0 and not reason, (code, reason)
