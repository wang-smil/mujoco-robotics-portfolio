"""Prepare the pinned upstream dependency and install the local environment extensions."""
import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
UPSTREAM = 'https://github.com/apirrone/Open_Duck_Playground.git'
REVISION = 'b9be205ac64488c23504ca42e5ec790337adeec3'
PACKAGE = Path('playground/open_duck_mini_v2')
REFERENCE_SHA = '5850c0610ed89e2860e7047f9ee27d8412462e1199752952c3c2a0efd2cb7a25'
JOYSTICK_SHA = '254b51e17c6e0aece2a6efe5921ba32cbf11234b9bf0194f17bbeed128df55f8'
AUTHOR_SHA = '3c606f9381a1710cc8fecdb7442787dcbfce3ee9bc02a6f1224774ab2b3a1067'
AUTHOR_URL = ('https://raw.githubusercontent.com/apirrone/Open_Duck_Mini/'
              'b23317a485b3cec7d8417f352478778b3475173c/BEST_WALK_ONNX_2.onnx')


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def adapt_joystick(path):
    source = path.read_text(encoding='utf-8')
    source = source.replace('    cost_stand_still,\n', '    cost_stand_still,\n    reward_feet_air_time,\n', 1)
    replacements = {
        'tracking_lin_vel=': 'tracking_lin_vel=15.0,  # 原 2.5：让速度跟踪主导',
        'tracking_ang_vel=': 'tracking_ang_vel=10.0,  # 原 6.0',
        'stand_still=': 'stand_still=-5.0,  # 命令非零但速度≈0 时罚（原 -0.2 恒不触发）\n'
                       '                feet_air_time=10.0,  # 官方抬脚奖励，鼓励迈步',
        'alive=': 'alive=4.0,  # 原 20：压低存活权重，让走路胜出',
        'imitation=': 'imitation=5.0,  # 原 1.0：参考步态主导',
    }
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(' ' * 16):
            for key, replacement in replacements.items():
                if line[16:].startswith(key):
                    lines[i] = ' ' * 16 + replacement + '\n'
    source = ''.join(lines)
    start = source.index('            "stand_still": cost_stand_still(')
    end = source.index('            ),\n', start) + len('            ),\n')
    source = source[:start] + '''            "stand_still": jp.exp(
                -jp.linalg.norm(self.get_local_linvel(data)[:2]) / 0.02
            ) * (jp.linalg.norm(info["command"][:2]) > 0.05),
            "feet_air_time": reward_feet_air_time(
                info["feet_air_time"],
                first_contact,
                info["command"],
            ),
''' + source[end:]
    result = source.encode('utf-8')
    if hashlib.sha256(result).hexdigest() != JOYSTICK_SHA:
        raise ValueError('Adapted joystick bytes differ from the training manifest')
    path.write_bytes(result)


def prepare(source):
    vendor = ROOT / 'third_party/Open_Duck_Playground'
    working = ROOT / 'duck_training'
    if working.exists():
        raise FileExistsError(f'Training tree already exists at {working}; use a new checkout to prepare it again')
    vendor.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.duck-setup-', dir=ROOT) as temporary:
        temp = Path(temporary)
        if not vendor.exists():
            repository = temp / 'upstream'
            repository.mkdir()
            git(repository, 'init', '-q')
            git(repository, 'fetch', '-q', '--depth=1', source, REVISION)
            git(repository, 'checkout', '-q', '--detach', 'FETCH_HEAD')
            repository.rename(vendor)
        if git(vendor, 'rev-parse', 'HEAD') != REVISION:
            raise ValueError(f'Unexpected upstream commit in {vendor}')
        staging = temp / 'training'
        shutil.copytree(vendor / 'playground', staging / 'playground', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT / 'extensions', staging, dirs_exist_ok=True)
        reference = staging / PACKAGE / 'data/polynomial_coefficients.pkl'
        if hashlib.sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA:
            raise ValueError('Upstream reference-motion file failed its checksum')
        adapt_joystick(staging / PACKAGE / 'joystick.py')
        staging.rename(working)


def author_policy():
    output = ROOT / 'references/author_walk_policy.onnx'
    if output.is_file():
        data = output.read_bytes()
    else:
        with urlopen(AUTHOR_URL, timeout=40) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != AUTHOR_SHA:
        raise ValueError('Author ONNX checksum mismatch')
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', default=UPSTREAM, help='Git URL or local mirror containing the pinned commit')
    parser.add_argument('--author-policy', action='store_true', help='Also fetch the pinned author ONNX')
    parser.add_argument('--author-only', action='store_true', help='Fetch the author ONNX for an already prepared checkout')
    args = parser.parse_args()
    if not args.author_only:
        prepare(args.upstream)
    if args.author_policy or args.author_only:
        author_policy()


if __name__ == '__main__':
    main()
