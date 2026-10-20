#!/usr/bin/env python3
"""按既有交付布局组装 Windows 构建产物；只写新目录，不复用旧业务二进制。"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('bin', 'obj', '__pycache__', '*.pyc'))
    else:
        shutil.copy2(src, dst)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def finalize(out):
    for required in ['package_metadata.json','reports/windows_validation.md','reports/validation.json','README-项目说明.md','使用说明.md']:
        if not (out/required).is_file(): raise FileNotFoundError(out/required)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name != '文件清单.md')
    files.append('文件清单.md')
    files.sort()
    (out/'文件清单.md').write_text('# 文件清单\n\n本目录为 C# 整机 DLL + ONNX Windows x64 交付。\n\n共 '+str(len(files))+' 个文件：\n\n'+'\n'.join(f'- `{p}`' for p in files)+'\n', encoding='utf-8')
    # 清单自身不自引用哈希；外置校验文件覆盖所有包内文件。
    manifest = {p: hashlib.sha256((out/p).read_bytes()).hexdigest() for p in files}
    write_json(out.with_suffix('.manifest.json'), manifest)
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(out/p, out.name+'/'+p)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for p, digest in manifest.items():
            assert hashlib.sha256(z.read(out.name+'/'+p)).hexdigest() == digest
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({'directory':str(out), 'files':len(files), 'sha256':digest}, ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prebuild', type=Path, required=True)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--windows-work', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--finalize-only', action='store_true')
    a=ap.parse_args(); out=a.output.resolve()
    if a.finalize_only:
        finalize(out); return
    if out.exists() or out.with_suffix('.zip').exists(): raise FileExistsError(out)
    pre=a.prebuild; old=a.template; work=a.windows_work; build=work/'build'
    checks=json.loads((a.evidence/'windows_validation.json').read_text(encoding='utf-8-sig'))
    assert len(checks)>=9 and all(x['exit_code']==x['expected'] for x in checks)
    out.mkdir(parents=True)
    copy(build/'verify',out/'verify')
    copy(build/'winforms',out/'examples/WinFormsDemo/publish')
    copy(pre/'deployment/dotnet/examples',out/'examples/config-staging')
    for p in (out/'examples/config-staging').iterdir(): p.rename(out/'examples'/p.name)
    (out/'examples/config-staging').rmdir()
    copy(pre/'deployment/dotnet/samples/WinFormsDemo',out/'examples/WinFormsDemo/source')
    csproj=out/'examples/WinFormsDemo/source/WinFormsDemo.csproj'
    text=csproj.read_text().replace('<ProjectReference Include="../../src/Dehumidifier.Simulation/Dehumidifier.Simulation.csproj" />','<PackageReference Include="Dehumidifier.Simulation" Version="1.0.1" />').replace('../../examples/*.json','../../*.json')
    csproj.write_text(text,encoding='utf-8')
    copy(pre/'deployment/dotnet/src/Dehumidifier.Simulation/models',out/'models')
    for p in (old/'lib').iterdir(): copy(build/'verify'/p.name,out/'lib'/p.name)
    for p in (old/'runtimes').rglob('*'):
        if p.is_file(): copy(build/'verify'/p.name,out/'runtimes'/p.relative_to(old/'runtimes'))
    copy(old/'nuget',out/'nuget')
    # 旧交付的同名主包不能进入新目录，否则离线源可能继续解析到旧 DLL。
    for p in (out/'nuget').glob('Dehumidifier.Simulation.*.nupkg'):
        p.unlink()
    for p in (build/'nuget').glob('*.nupkg'): shutil.copy2(p,out/'nuget'/p.name)
    copy(work/'source/deployment',out/'source/deployment')
    for name in ['README.md','AGENTS.md']: copy(pre/name,out/'source'/name)
    copy(pre/'docs',out/'source/docs')
    copy(pre/'results/simulation/python_reference',out/'results/simulation/python_reference')
    for case in ['qy1070_case','default_case','unconverged_max_iter_1']:
        records=list((build/'win-results-verified'/case).rglob('result.json'))
        assert len(records)==1, (case,records)
        copy(records[0].parent,out/'results/simulation/windows'/case)
    copy(pre/'reports/onnx_export/model_contract.json',out/'reports/onnx_export_contract.json')
    copy(pre/'reports/onnx_export/preprocessing.json',out/'reports/preprocessing_export.json')
    copy(a.evidence/'windows_validation.json',out/'reports/windows_validation.json')
    (out/'reports/build.log').write_text((a.evidence/'build-final.log').read_text(encoding='utf-16'), encoding='utf-8')
    copy(old/'THIRD-PARTY-NOTICES.md',out/'THIRD-PARTY-NOTICES.md')
    copy(old/'dependencies.json',out/'dependencies.json')
    docs=Path(__file__).resolve().parent/'docs'
    for name in ['README-项目说明.md','使用说明.md']: copy(docs/name,out/name)
    shutil.copy2(docs/'离线还原说明.md',out/'nuget/离线还原说明.md')
    copy(a.evidence/'windows_validation.md',out/'reports/windows_validation.md')
    comparison = subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'validate_migration.py'),'--python-root',str(out/'results/simulation/python_reference'),'--csharp-root',str(out/'results/simulation/windows'),'--output',str(out/'reports/validation.json')],check=False)
    if comparison.returncode not in (0, 1) or not (out/'reports/validation.json').exists():
        raise RuntimeError('迁移对比未产生有效报告')
    # 确认所有嵌入模型均为本次源包，业务 DLL 只来自本次 Windows 构建。
    for p in (out/'models').iterdir():
        for base in ['verify/models','examples/WinFormsDemo/publish/models']:
            assert (out/base/p.name).read_bytes()==p.read_bytes(), (base,p.name)
    for base in ['lib','examples/WinFormsDemo/publish']:
        assert (out/base/'Dehumidifier.Simulation.dll').read_bytes()==(out/'verify/Dehumidifier.Simulation.dll').read_bytes()
    assert (out/'lib/Dehumidifier.Simulation.dll').read_bytes()!=(old/'lib/Dehumidifier.Simulation.dll').read_bytes()
    for case in ['qy1070_case','default_case']: assert (out/'examples/WinFormsDemo/publish/examples'/f'{case}.json').exists()
    write_json(out/'package_metadata.json',{'package_name':out.name,'created_at':datetime.now().astimezone().isoformat(),'models':['models/xf.onnx','models/main.onnx'],'source_prebuild':pre.name,'layout_template':old.name,'build_evidence':'reports/windows_validation.md','numeric_evidence':'reports/validation.md','engineering_acceptance':'not_evaluated','verified_on':'Windows x64','documentation_status':'pending'})
    print(out)

if __name__=='__main__': main()
