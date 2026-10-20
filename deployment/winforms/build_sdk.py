"""从当前仓库制作独立开发 SDK；每次使用新的 dist 子目录。"""
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def build():
    import yaml
    name = 'dehumidifier_winforms_sdk_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target = ROOT / 'dist' / name
    target.mkdir(parents=True, exist_ok=False)
    template = ROOT / 'deployment/winforms'
    for source in template.rglob('*'):
        if source.is_file() and source.suffix in {'.py', '.ps1', '.cs', '.csproj', '.txt', '.md'} and source.name != 'build_sdk.py':
            destination = target / source.relative_to(template)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for source in (ROOT / 'simulation/dual_rotor_sim').rglob('*.py'):
        destination = target / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for profile in ('xf', 'main'):
        package = ROOT / 'artifacts' / f'kan_{profile}_single'
        provenance = json.loads((package / 'training_run.json').read_text(encoding='utf-8'))
        for relative, expected in provenance['sha256'].items():
            if hashlib.sha256((package / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'运行时文件与训练来源不一致：{profile}/{relative}')
        for relative in ('model/best_model.pt', 'model/scalers.pkl', 'model/scaler_meta.json', 'model/metadata.json',
                         'model_metadata.json', 'config_snapshot.yaml', 'training_run.json'):
            destination = target / 'artifacts' / package.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package / relative, destination)
    baselines = {
        'qy1070_case': '20260909_093333_200046_qy1070_case',
        'default_case': '20260909_093345_090568_default_case',
    }
    for case, run in baselines.items():
        config = ROOT / 'simulation/configs' / (case + '.yaml')
        document = json.loads((ROOT / 'results/simulation' / run / 'result.json').read_text(encoding='utf-8'))
        parsed = yaml.safe_load(config.read_text(encoding='utf-8'))
        if parsed != document['config']:
            raise ValueError(f'当前配置与参考结果不一致：{case}')
        write_json(target / 'examples' / (case + '.request.json'),
                   dict(protocol_version=1, id=case, command='simulate', config=parsed))
        destination = target / 'examples/configs' / config.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, destination)
        for filename in ('result.json', 'fig1_dashboard.png', 'diagnostics.json', 'model_sources.json'):
            source = ROOT / 'results/simulation' / run / filename
            destination = target / 'examples/results/simulation' / case / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(ROOT / 'results/simulation/20260909_kan_tuning/调试报告.md', target / '模型联调报告.md')
    write_json(target / 'package_info.json', {
        'name': name, 'created_at': datetime.now().astimezone().isoformat(),
        'delivery_type': '源码与模型 SDK；Windows EXE 由开发人员原生构建',
        'protocol_version': 1, 'python': sys.version, 'platform': platform.platform(),
        'runtime_versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'numpy', 'pandas', 'scikit-learn', 'PyYAML', 'matplotlib', 'openpyxl')},
        'windows_build_verified': False, 'csharp_compiled': False,
        'baselines': baselines,
        'limitations': '稳态调试模型，域外输入与水分收支偏差尚未解决；不代表工程验收。',
    })
    hashes = {p.relative_to(target).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(target.rglob('*')) if p.is_file()}
    write_json(target / 'SHA256SUMS.json', hashes)
    archive = shutil.make_archive(str(target), 'zip', target.parent, target.name)
    print(target)
    print(archive)
    return target


if __name__ == '__main__':
    build()
