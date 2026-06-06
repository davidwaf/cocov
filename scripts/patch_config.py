"""
scripts/patch_config.py
-----------------------
Patches config.yaml encoder section for a given backbone.
Called by run_encoder.sh before each pipeline run.

Usage:
    python scripts/patch_config.py ENCODER ENC_CODE INPUT_SIZE [CONFIG_PATH]

Example:
    python scripts/patch_config.py arcface_r100 ENC03_ARCFACE_R100 112
"""

import sys
import re
from pathlib import Path

VALID = {
    'facenet':         ('ENC01_FACENET',       160),
    'arcface_r50':     ('ENC02_ARCFACE_R50',   112),
    'arcface_r100':    ('ENC03_ARCFACE_R100',  112),
    'adaface':         ('ENC04_ADAFACE',       112),
    'vitb_arcface':    ('ENC05_VITB_ARCFACE',  112),
    'mobilefacenet':   ('ENC04_MOBILEFACENET', 112),
}

def patch(encoder: str, config_path: str = 'config/config.yaml') -> None:
    if encoder not in VALID:
        raise ValueError(
            f"Unknown encoder '{encoder}'. "
            f"Valid: {list(VALID.keys())}"
        )
    code, size = VALID[encoder]
    path = Path(config_path)
    src  = path.read_text()

    # Replace each field individually with a tight pattern
    src = re.sub(r'(?m)(^  name:\s*")[^"]+(")',
                 rf'\g<1>{encoder}\g<2>', src)
    src = re.sub(r'(?m)(^  code:\s*")[^"]+(")',
                 rf'\g<1>{code}\g<2>', src)
    src = re.sub(r'(?m)(^  input_size:\s*)\d+',
                 rf'\g<1>{size}', src)
    src = re.sub(r'(?m)(^  cache_subdir:\s*")[^"]+(")',
                 rf'\g<1>{encoder}\g<2>', src)

    path.write_text(src)

    # Verify
    import yaml
    cfg = yaml.safe_load(path.read_text())['encoder']
    assert cfg['name']         == encoder, f"name: {cfg['name']}"
    assert cfg['code']         == code,    f"code: {cfg['code']}"
    assert cfg['input_size']   == size,    f"input_size: {cfg['input_size']}"
    assert cfg['cache_subdir'] == encoder, f"cache_subdir: {cfg['cache_subdir']}"

    print(f"config.yaml patched:")
    print(f"  name        = {cfg['name']}")
    print(f"  code        = {cfg['code']}")
    print(f"  input_size  = {cfg['input_size']}")
    print(f"  cache_subdir= {cfg['cache_subdir']}")

if __name__ == '__main__':
    enc    = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else 'config/config.yaml'
    patch(enc, config)
