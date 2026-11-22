"""Download a pinned public policy snapshot and verify every file against Hub metadata."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'

from huggingface_hub import HfApi, snapshot_download

REPO = 'lerobot/pi0_libero_finetuned_v044'
REVISION = '45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f'
DEST = Path('/root/autodl-tmp/models/pi0_libero_finetuned_v044')
RECORDS = Path('/root/autodl-tmp/lerobot-env-records')
status_file = RECORDS / 'model-download-status.json'


def status(state, **details):
    """Persist download progress without writing authentication data."""
    status_file.write_text(json.dumps({'state': state, 'repo_id': REPO, 'revision': REVISION,
                                       'destination': str(DEST), **details}, indent=2))


try:
    status('fetching_metadata')
    api = HfApi(endpoint=os.environ['HF_ENDPOINT'], token=False)
    info = api.model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION, info.sha
    entries = []
    for item in info.siblings:
        lfs = item.lfs
        entries.append({'path': item.rfilename, 'size': item.size,
                        'sha256': lfs.sha256 if lfs else None, 'git_blob_sha1': item.blob_id})
    total = sum(item['size'] or 0 for item in entries)
    manifest = {'repo_id': REPO, 'revision': REVISION, 'files': entries, 'total_bytes': total}
    (RECORDS / 'model-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    status('downloading', total_bytes=total, file_count=len(entries))
    start = time.monotonic()
    if not all((DEST / entry['path']).exists() and (DEST / entry['path']).stat().st_size == entry['size'] for entry in entries):
        snapshot_download(REPO, revision=REVISION, local_dir=str(DEST), max_workers=2, token=False)
    status('verifying', total_bytes=total, file_count=len(entries))
    for item in entries:
        path = DEST / item['path']
        size = path.stat().st_size
        assert size == item['size'], (item['path'], size, item['size'])
        digest = hashlib.sha256() if item['sha256'] else hashlib.sha1()
        if not item['sha256']:
            digest.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        expected = item['sha256'] or item['git_blob_sha1']
        assert digest.hexdigest() == expected, f'Checksum mismatch: {item["path"]}'
        print(f'Verified: {item["path"]} ({size} bytes)', flush=True)
    status('complete', total_bytes=total, file_count=len(entries), elapsed_seconds=time.monotonic() - start)
    print('DOWNLOAD AND CHECKSUM VERIFICATION COMPLETE', flush=True)
except Exception as error:
    status('failed', error=f'{type(error).__name__}: {error}')
    raise
