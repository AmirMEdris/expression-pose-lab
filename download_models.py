"""Fetch pinned model artifacts; verify checksums before installing them."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / 'models'
ASSETS = {
    'expression-baseline.h5': (
        'https://raw.githubusercontent.com/AmirMEdris/Facial-Expression-Recognition/514316f2e5dce38a0e8ab35b50807f44fc57a2a4/model_filter.h5',
        '869f55275072fa5e4523767ef912d48c82289570ab6f19c8f9a512a86ad6f2b8'),
    'pose_landmarker_lite.task': (
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'),
}


def checksum(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        target = ROOT / name
        if target.exists() and checksum(target) == expected:
            print(f'Verified {name}')
            continue
        temporary = target.with_suffix(target.suffix + '.download')
        try:
            print(f'Downloading {name}')
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            if checksum(temporary) != expected:
                raise RuntimeError(f'Checksum mismatch for {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
