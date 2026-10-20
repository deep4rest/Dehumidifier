"""供桌面宿主使用的 UTF-8 JSON Lines 接口；一个进程串行复用两个 KAN。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import traceback

PROTOCOL_VERSION = 1
LIMITATIONS = [
    '当前为稳态模型；不描述启动过程或转轮蓄热动态。',
    '当前两参考工况存在训练范围外输入及水分收支偏差；数值收敛不代表工程物理验收。详见交付说明。',
    'cooling_load/heating_load 是热负荷，不是整机电耗。',
]


def _response(request_id, status, code, message, **extra):
    return dict(protocol_version=PROTOCOL_VERSION, id=request_id, status=status,
                code=code, message=message, **extra)


def _validate(request):
    if not isinstance(request, dict):
        raise ValueError('请求必须为 JSON 对象')
    if type(request.get('protocol_version')) is not int or request['protocol_version'] != 1:
        raise ValueError('protocol_version 必须为整数 1')
    if not isinstance(request.get('id'), str) or not 0 < len(request['id']) <= 128:
        raise ValueError('id 必须为 1 至 128 个字符的字符串')
    command = request.get('command')
    if command not in ('ping', 'simulate'):
        raise ValueError('command 仅支持 ping 或 simulate')
    allowed = {'protocol_version', 'id', 'command'}
    if command == 'simulate':
        allowed |= {'config', 'solver'}
    if set(request) - allowed:
        raise ValueError('请求包含未知字段')
    solver = request.get('solver', {})
    if not isinstance(solver, dict) or set(solver) - {'max_iter', 'relax', 'tol_T_degC', 'tol_W_g_per_kg'}:
        raise ValueError('solver 必须是仅含已定义求解参数的对象')
    if 'max_iter' in solver and (type(solver['max_iter']) is not int or not 1 <= solver['max_iter'] <= 10000):
        raise ValueError('max_iter 必须是 1 至 10000 的整数')
    for key in ('relax', 'tol_T_degC', 'tol_W_g_per_kg'):
        if key in solver:
            value = solver[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} 必须是有限正数')
    if solver.get('relax', 0.8) > 1:
        raise ValueError('relax 不得大于 1')
    return command, solver


class Engine:
    def __init__(self, artifacts_dir: Path, results_dir: Path):
        self.artifacts_dir = artifacts_dir.resolve()
        self.results_dir = results_dir.resolve()
        self.system = None

    def handle(self, request):
        request_id = request.get('id') if isinstance(request, dict) and isinstance(request.get('id'), str) else None
        try:
            command, solver = _validate(request)
            runtime = None
            if command == 'simulate':
                from .config import parse_simulation_config, build_runtime_inputs
                runtime = build_runtime_inputs(parse_simulation_config(request.get('config')))
        except (ValueError, TypeError, OverflowError) as exc:
            return _response(request_id, 'error', 'INVALID_REQUEST', str(exc))
        try:
            if self.system is None:
                from . import build_system
                self.system = build_system(self.artifacts_dir, device='cpu')
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _response(request_id, 'error', 'MODEL_LOAD_FAILED', f'模型加载失败：{exc}')
        if command == 'ping':
            return _response(request_id, 'ok', 'OK', '两个 KAN 已加载，可执行仿真', warnings=LIMITATIONS)
        try:
            result = self.system.simulate_coupled_xf(
                runtime['oa'], runtime['ra'], runtime['regen_oa'],
                runtime['params'], runtime['setpoints'], **solver)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _response(request_id, 'error', 'SIMULATION_FAILED', f'仿真失败：{exc}')
        try:
            from .run_records import save_simulation_run, result_to_dict
            run_dir = save_simulation_run(result, request['config'], Path('winforms_request.json'), self.results_dir)
            payload = result_to_dict(result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _response(request_id, 'error', 'SAVE_FAILED', f'结果留档失败：{exc}')
        converged = result.convergence.get('converged') is True
        return _response(
            request_id, 'ok' if converged else 'unconverged',
            'OK' if converged else 'NOT_CONVERGED',
            '仿真完成' if converged else '迭代未收敛，已保存调试结果',
            result=payload, warnings=LIMITATIONS,
            files={'result_json': str(run_dir / 'result.json'),
                   'dashboard_png': str(run_dir / 'fig1_dashboard.png')})


def _reject_constant(value):
    raise ValueError(f'JSON 不允许 {value}')


def main():
    parser = argparse.ArgumentParser(description='WinForms 整机计算引擎，UTF-8 单行 JSON 协议')
    parser.add_argument('--artifacts-dir', type=Path, required=True, help='两个 KAN 包的父目录')
    parser.add_argument('--results-dir', type=Path, required=True, help='有写入权限的仿真结果根目录')
    parser.add_argument('--once', action='store_true', help='读取一个请求后退出；成功/失败/未收敛退出码为 0/2或3/4')
    args = parser.parse_args()
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(args.results_dir.resolve().parent / '.cache' / 'matplotlib'))
    engine = Engine(args.artifacts_dir, args.results_dir)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line, parse_constant=_reject_constant)
            # 依赖库的意外打印也不能污染宿主读取的协议。
            with redirect_stdout(sys.stderr):
                reply = engine.handle(request)
        except (ValueError, TypeError) as exc:
            reply = _response(None, 'error', 'INVALID_REQUEST', f'请求解析失败：{exc}')
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = _response(None, 'error', 'INTERNAL_ERROR', f'引擎异常：{exc}')
        print(json.dumps(reply, ensure_ascii=False, allow_nan=False), flush=True)
        if args.once:
            return 4 if reply['status'] == 'unconverged' else (3 if reply['code'] == 'SAVE_FAILED' else 2 if reply['status'] == 'error' else 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
