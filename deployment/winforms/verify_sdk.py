"""在解压后的 SDK 中校验文件与完整计算结果；可指定 Windows EXE。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime


def compare(actual, expected, path='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f'{path} 字段不一致')
        for key in expected:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f'{path} 长度不一致')
        for i, value in enumerate(expected):
            compare(actual[i], value, f'{path}[{i}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual != expected:
        raise AssertionError(f'{path}: {actual} != {expected}')


def main():
    parser = argparse.ArgumentParser(description='开发交付包完整性和双工况对照')
    parser.add_argument('--engine', type=Path, help='Windows EXE，省略则使用当前 Python')
    parser.add_argument('--results-dir', type=Path, help='有写入权限的验证结果目录')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    hashes = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, expected in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f'文件校验失败：{name}')
    output = (args.results_dir or root / 'results' / 'simulation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_sdk_verify')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(args.engine.resolve())] if args.engine else [sys.executable, '-I', str(root / 'engine.py')]
    requests = [json.loads((root / 'examples' / f'{name}.request.json').read_text(encoding='utf-8')) for name in ('qy1070_case', 'default_case')]
    ping = {'protocol_version': 1, 'id': 'ping', 'command': 'ping'}
    bad = {'protocol_version': 1, 'id': 'bad', 'command': 'simulate', 'config': {}}
    short = {**requests[0], 'id': 'unconverged', 'solver': {'max_iter': 1}}
    wire = '\n'.join(json.dumps(x, ensure_ascii=False) for x in [ping, bad, *requests, short]) + '\n'
    # 不使用调用者工作目录或 PYTHONPATH；源码启动器明确选择交付包内的仿真包。
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    with tempfile.TemporaryDirectory(prefix='dehumidifier-cwd-') as cwd:
        proc = subprocess.run(command + ['--results-dir', str(output)], input=wire,
                              capture_output=True, text=True, encoding='utf-8', cwd=cwd, env=env, timeout=180)
    (output / 'stderr.log').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'引擎退出 {proc.returncode}：{proc.stderr}')
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(replies) != 5 or replies[0]['status'] != 'ok' or replies[1]['code'] != 'INVALID_REQUEST':
        raise AssertionError('协议或错误恢复失败')
    for name, reply in zip(('qy1070_case', 'default_case'), replies[2:4]):
        if reply['status'] != 'ok':
            raise AssertionError(reply)
        expected = json.loads((root / 'examples/results/simulation' / name / 'result.json').read_text(encoding='utf-8'))['result']
        # 收敛历史允许跨平台末位差异/迭代数差异；最终状态、负荷和转轮输出逐字段比较。
        compare({k: v for k, v in reply['result'].items() if k != 'convergence'},
                {k: v for k, v in expected.items() if k != 'convergence'})
        if reply['result']['convergence']['converged'] is not True:
            raise AssertionError('参考工况未收敛')
    if replies[-1]['status'] != 'unconverged':
        raise AssertionError('未收敛未正确上报')
    for reply in replies[2:]:
        record = Path(reply['files']['result_json'])
        png = Path(reply['files']['dashboard_png'])
        if not record.is_file() or not png.is_file() or record.parent != png.parent:
            raise AssertionError('JSON/PNG 留档不完整')
        if json.loads(record.read_text(encoding='utf-8'))['result'] != reply['result']:
            raise AssertionError('接口与落盘结果不一致')
    report = {'status': 'passed', 'platform': sys.platform, 'python': sys.version,
              'engine': command, 'checked_files': len(hashes), 'responses': replies,
              'windows_exe_tested': bool(args.engine and sys.platform == 'win32'),
              'comparison': {'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-5}}
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'校验通过：{len(hashes)} 个文件、双工况完整状态/负荷、错误恢复和未收敛留档。记录：{output}')


if __name__ == '__main__':
    main()
