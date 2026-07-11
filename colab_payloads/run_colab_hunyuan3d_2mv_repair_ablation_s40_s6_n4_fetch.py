# Paste this into one Colab Python cell to download and launch the packaged benchmark.
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import urllib.request

PAYLOAD_URL = "https://raw.githubusercontent.com/DrStrangel0ve/3dprintpic/codex-colab-payloads/colab_payloads/modelnet10_hunyuan3d_2mv_repair_ablation_s40_s6_n4_colab_inputs.tar.gz"
ARCHIVE_PATH = pathlib.Path("/content/modelnet10_hunyuan3d_2mv_repair_ablation_s40_s6_n4_colab_inputs.tar.gz")
EXTRACT_ROOT = pathlib.Path("/content/3dprintpic_colab_inputs/modelnet10_hunyuan3d_2mv_repair_ablation_s40_s6_n4")
EXPECTED_SHA256 = "3746789046291634300864c2e29e221cf93afd49d5ca17346549afaee53633fa"
EXPECTED_SIZE = 806388
LAUNCH_ENV = {}

ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f'Downloading {PAYLOAD_URL}')
with urllib.request.urlopen(PAYLOAD_URL) as response:
    payload = response.read()
ARCHIVE_PATH.write_bytes(payload)

actual_size = ARCHIVE_PATH.stat().st_size
actual_sha256 = hashlib.sha256(payload).hexdigest()
print({'archive': str(ARCHIVE_PATH), 'bytes': actual_size, 'sha256': actual_sha256})
if actual_size != EXPECTED_SIZE:
    raise SystemExit(f'archive size mismatch: expected {EXPECTED_SIZE} got {actual_size}')
if actual_sha256 != EXPECTED_SHA256:
    raise SystemExit(f'archive sha256 mismatch: expected {EXPECTED_SHA256} got {actual_sha256}')

EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
run_script = EXTRACT_ROOT / 'run_colab_eval.sh'
with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:
    try:
        member = tar.getmember('run_colab_eval.sh')
    except KeyError as exc:
        raise SystemExit('archive does not contain run_colab_eval.sh; regenerate with --include-run-script') from exc
    extracted = tar.extractfile(member)
    if extracted is None:
        raise SystemExit('could not read run_colab_eval.sh from archive')
    run_script.write_bytes(extracted.read())
run_script.chmod(0o755)

env = os.environ.copy()
env['EXTRACT_ROOT'] = str(EXTRACT_ROOT)
env['EXPECTED_SHA256'] = EXPECTED_SHA256
for key, value in LAUNCH_ENV.items():
    env[key] = value
if LAUNCH_ENV:
    print({'launch_env': LAUNCH_ENV})
print(f'Launching {run_script} with {ARCHIVE_PATH}')
completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)

def _colab_sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _print_colab_result_summary():
    summary_path = EXTRACT_ROOT / 'results_summary.json'
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            print('---RESULTS_SUMMARY_JSON---')
            print(json.dumps(summary, indent=2, sort_keys=True))
        except Exception as exc:
            print({'results_summary_read_error': f'{type(exc).__name__}: {exc}', 'path': str(summary_path)})
    else:
        print({'results_summary_missing': str(summary_path)})
    archive_candidates = []
    for archive_key in ('results_compact_archive', 'results_archive', 'result_archive'):
        archive_value = summary.get(archive_key)
        if archive_value:
            archive_candidates.append(pathlib.Path(archive_value))
    if not archive_candidates:
        archive_candidates.extend(sorted(pathlib.Path('/content').glob('*results*.tar.gz')))
    seen = set()
    archives = []
    for archive in archive_candidates:
        archive = pathlib.Path(archive)
        key = str(archive)
        if key in seen or not archive.exists():
            continue
        seen.add(key)
        archives.append({'path': key, 'bytes': archive.stat().st_size, 'sha256': _colab_sha256_file(archive)})
    print('---RESULT_ARCHIVES_JSON---')
    print(json.dumps(archives, indent=2, sort_keys=True))

_print_colab_result_summary()
completed.check_returncode()
