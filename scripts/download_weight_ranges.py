"""Download the large public weight in resumable HTTP ranges, with bounded memory."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import requests
from urllib.parse import urlsplit

records = Path('/root/autodl-tmp/lerobot-env-records')
manifest = json.loads((records / 'model-manifest.json').read_text())
dest = Path('/root/autodl-tmp/models/pi0_libero_finetuned_v044')
entry = next(item for item in manifest['files'] if item['path'] == 'model.safetensors')
size = entry['size']
chunk_size = 64 * 1024 * 1024
parts = dest / '.download-parts'
parts.mkdir(exist_ok=True)
url = f"https://hf-mirror.com/{manifest['repo_id']}/resolve/{manifest['revision']}/model.safetensors"
for resolve_attempt in range(8):
    resolution = requests.head(url, allow_redirects=True, timeout=(15, 45))
    if resolution.status_code != 429:
        resolution.raise_for_status()
        url = resolution.url
        break
    time.sleep(15)
else:
    raise RuntimeError('Mirror rate limit persisted')
print(f'Download host: {urlsplit(url).hostname}', flush=True)


def download_part(index):
    """Resume one range and reject servers returning the wrong byte range."""
    start = index * chunk_size
    end = min(size, start + chunk_size) - 1
    target = parts / f'{index:04d}.part'
    expected = end - start + 1
    attempt = 0
    while True:
        have = target.stat().st_size if target.exists() else 0
        if have == expected:
            return index
        assert have < expected
        try:
            request_end = min(end, start + have + 8 * 1024 * 1024 - 1)
            with requests.get(url, headers={'Range': f'bytes={start + have}-{request_end}', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                assert response.status_code == 206, response.status_code
                actual = response.headers.get('Content-Range', '')
                assert actual.startswith(f'bytes {start + have}-{request_end}/'), actual
                with target.open('ab') as output:
                    for block in response.iter_content(1024 * 1024):
                        output.write(block)
            assert target.stat().st_size == request_end - start + 1
            attempt = 0
        except Exception as error:
            print(f'Retrying part {index}, attempt {attempt + 1}: {type(error).__name__}: {error}', flush=True)
            attempt += 1
            if attempt >= 6:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))


count = (size + chunk_size - 1) // chunk_size
with ThreadPoolExecutor(max_workers=64) as pool:
    futures = [pool.submit(download_part, index) for index in range(count)]
    for done, future in enumerate(as_completed(futures), 1):
        index = future.result()
        print(f'Completed {done}/{count} ranges (part {index})', flush=True)
temporary = dest / 'model.safetensors.assembling'
with temporary.open('wb') as output:
    for index in range(count):
        with (parts / f'{index:04d}.part').open('rb') as source:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
assert temporary.stat().st_size == size
temporary.replace(dest / 'model.safetensors')
print('RANGE DOWNLOAD COMPLETE; checksum verification required', flush=True)
