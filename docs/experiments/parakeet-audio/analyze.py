"""Summarize complete runs without changing the pinned scoring rules."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
name = sys.argv[1] if len(sys.argv) > 1 else 'screen'
folder = HERE / name
rows = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
groups = defaultdict(list)
for row in rows:
    groups[row['variant'], row['mode']].append(row)
result = []
for (variant, mode), items in groups.items():
    assert len(items) == len({r['file'] for r in items}) == 30
    for cohort in ['all', 'original', 'added']:
        subset = [r for r in items if cohort == 'all' or r['cohort'] == cohort]
        scores = {}
        for kind in ['normalized', 'strict']:
            counts = {k: sum(r[kind][k] for r in subset) for k in ['errors', 'reference_words', 'substitutions', 'deletions', 'insertions']}
            counts['wer'] = counts['errors'] / counts['reference_words']
            scores[kind] = counts
        # Whole mode's final entry is the second inference pass. Segmented mode's
        # latter half contains the second traversal. The first traversal warms inference.
        warm_ms = [sum(t[0] for t in r['timings_ms'][len(r['timings_ms']) // 2:]) for r in subset]
        result.append(dict(variant=variant, mode=mode, cohort=cohort, coverage=len(subset), **scores,
                           summed_second_pass_ms=sum(warm_ms), median_second_pass_ms=statistics.median(warm_ms),
                           peak_rss_kib=max(r['peak_rss_kib'] for r in subset),
                           wall_seconds=sum(r['wall_seconds'] for r in subset)))
(folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
base = {(r['mode'], r['file']): r for r in rows if r['variant'] == 'raw'}
with (folder / 'per-recording.tsv').open('w') as output:
    output.write('variant\tmode\tcohort\tfile\twords\tnormalized_errors\tnormalized_delta_vs_raw\tstrict_errors\tstrict_delta_vs_raw\tsecond_pass_ms\tpeak_rss_kib\twall_seconds\n')
    for r in rows:
        b = base.get((r['mode'], r['file']))
        delta = lambda kind: str(r[kind]['errors'] - b[kind]['errors']) if b else ''
        values = [r['variant'], r['mode'], r['cohort'], r['file'], r['normalized']['reference_words'],
                  r['normalized']['errors'], delta('normalized'), r['strict']['errors'], delta('strict'),
                  sum(t[0] for t in r['timings_ms'][len(r['timings_ms']) // 2:]), r['peak_rss_kib'], r['wall_seconds']]
        output.write('\t'.join(map(str, values)) + '\n')
for r in result:
    print(r['variant'], r['mode'], r['cohort'], f"{r['normalized']['errors']}/{r['normalized']['reference_words']}",
          'strict', r['strict']['errors'], 'ms', r['summed_second_pass_ms'], 'RSS MiB', round(r['peak_rss_kib'] / 1024, 1))
