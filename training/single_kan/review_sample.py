"""生成只读源文件的小样本审阅包，不执行全量清洗或训练。"""
from pathlib import Path
import hashlib
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
from .config import (DATA_DIR, TRAINING_RESULTS_DIR, EXPECTED_SOURCE_SHA256,
                     SOURCE_INPUT_COLUMNS_BY_PROFILE, SOURCE_OUTPUT_COLUMNS_BY_PROFILE,
                     SOURCE_TO_CANONICAL_BY_PROFILE)

NEW_SHA256 = '4bf827fba88347b84ed7a576d9b8757b1c692e024ef9aab2a867c99d1bc9d411'


def clean_sample(profile, frame):
    """按数值隔离风量异常及完全重复，保留完整删除证据。"""
    if profile not in ('xf', 'main'):
        raise ValueError('未知转轮类型')
    data = frame.copy()
    if not np.isfinite(data.to_numpy(dtype=float)).all() or data.ID.duplicated().any():
        raise ValueError('样本存在非有限数值或重复ID，需先审计')
    bad = pd.Series(False, index=data.index)
    if profile == 'xf':
        bad = (data.Label_React_CMH - data.TextBox_Purge_CMH - data.TextBox_OA_CMH).abs() > 1
    rejected = data.loc[bad].copy()
    rejected['删除原因'] = '再生风量与冷吹及后级风量之和差值超过1'
    rejected['保留ID'] = pd.NA
    remaining = data.loc[~bad].copy()
    columns = list(data.columns.drop('ID'))
    duplicate = remaining.duplicated(columns, keep='first')
    removed = remaining.loc[duplicate].copy()
    removed['删除原因'] = '除ID外全部字段完全重复'
    if len(removed):
        first = remaining.groupby(columns, sort=False, dropna=False)['ID'].transform('first')
        removed['保留ID'] = first.loc[duplicate]
    else:
        removed['保留ID'] = pd.Series(dtype='Int64')
    return remaining.loc[~duplicate].copy(), pd.concat([rejected, removed]).reset_index(drop=True)


def export_review(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for profile, filename, expected in [
        ('xf', '新风转轮数据集_NEW.csv', NEW_SHA256),
        ('main', '主转轮数据集.xls', EXPECTED_SOURCE_SHA256['main']),
    ]:
        print(f'读取并校验 {filename}', flush=True)
        content = (DATA_DIR / filename).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f'{filename} 源哈希不一致')
        technical = (pd.read_csv(io.BytesIO(content), low_memory=False, encoding='utf-8-sig')
                     if profile == 'xf' else pd.read_excel(io.BytesIO(content), sheet_name='CollectRecord'))
        data = technical.iloc[1:].apply(pd.to_numeric, errors='raise').reset_index(drop=True)
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError('源数据存在非有限值')
        data['ID'] = data.ID.astype('int64')
        # 全表只读筛选审阅样本：每个几何组均匀取8行，补充边界和问题记录。
        selected = set()
        for _, group in data.groupby(['TextBox_Rotor_depth', 'TextBox_Rotor_diameter']):
            selected.update(group.index[np.linspace(0, len(group)-1, min(8,len(group)), dtype=int)])
        for col in data.columns.drop('ID'):
            selected.update([data[col].idxmin(), data[col].idxmax()])
        selected.update(data.index[data.duplicated(list(data.columns.drop('ID')), keep=False)])
        if profile == 'xf':
            selected.update(data.index[(data.Label_React_CMH-data.TextBox_Purge_CMH-data.TextBox_OA_CMH).abs()>1])
            selected.update(data.index[data.ID.eq(1407)])
        sample = data.loc[sorted(selected)].copy()
        clean, removed = clean_sample(profile, sample)
        dest = output / profile
        dest.mkdir()
        sample.to_csv(dest/'清洗前样本.csv', index=False, encoding='utf-8-sig')
        removed.to_csv(dest/'删除清单.csv', index=False, encoding='utf-8-sig')
        inputs = list(SOURCE_INPUT_COLUMNS_BY_PROFILE[profile])
        targets = list(SOURCE_OUTPUT_COLUMNS_BY_PROFILE[profile])
        mapping = SOURCE_TO_CANONICAL_BY_PROFILE[profile]
        X = clean[inputs].rename(columns=mapping)
        y = clean[targets].rename(columns=mapping)
        X.to_csv(dest/'X.csv', index=False, encoding='utf-8-sig')
        y.to_csv(dest/'y.csv', index=False, encoding='utf-8-sig')
        audit = clean.drop(columns=inputs+targets).rename(columns=mapping).copy()
        audit.insert(0,'样本行号',np.arange(1,len(clean)+1))
        audit['源文件行号'] = clean.index + 3
        audit['源文件'] = filename
        audit['源SHA256'] = actual
        audit.to_csv(dest/'audit.csv',index=False,encoding='utf-8-sig')
        review = pd.concat([clean[['ID']],X,y],axis=1)
        review.to_csv(dest/'清洗后审阅.csv',index=False,encoding='utf-8-sig')
        np.savez(dest/'dataset.npz',X=X.to_numpy(dtype=np.float32),y=y.to_numpy(dtype=np.float32),source_id=clean.ID.to_numpy(dtype=np.int64))
        stats = pd.concat([X,y],axis=1).agg(['min','max','mean','std']).T
        stats.to_csv(dest/'字段统计.csv',encoding='utf-8-sig')
        schema = {'profile':profile,'input_columns':list(X.columns),'output_columns':list(y.columns),
                  'dtype':'float32','scaled':False,'split':'未划分，仅供审阅',
                  'source_sha256':actual,'source_file':filename,'source_rows':len(data),
                  'sample_before':len(sample),'sample_after':len(clean),'removed_ids':removed.ID.tolist(),
                  'geometry_groups':int(clean.groupby(['TextBox_Rotor_depth','TextBox_Rotor_diameter']).ngroups)}
        (dest/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
        # 回读检查列顺序、行对齐和float32存储，不训练模型。
        arrays = np.load(dest/'dataset.npz')
        assert arrays['X'].shape == (len(clean),len(inputs))
        assert arrays['y'].shape == (len(clean),6)
        np.testing.assert_array_equal(arrays['source_id'],pd.read_csv(dest/'audit.csv').ID)
        np.testing.assert_allclose(pd.read_csv(dest/'X.csv'),X)
        np.testing.assert_allclose(pd.read_csv(dest/'y.csv'),y)
        reports[profile] = schema
        print(f'{profile}：{len(sample)} → {len(clean)}，删除ID {removed.ID.tolist()}',flush=True)
    (output/'审阅说明.md').write_text(
        '# KAN清洗小样本审阅\n\n新风采用已确认的NEW CSV，主转轮采用XLS；源文件只读且哈希已核验。'
        '\n\n每组均匀抽样，补充字段极值、全部完全重复记录和已知风量异常；这是刻意覆盖问题的审阅集，不能用删除比例估算全量异常率。'
        '\n\n优先查看各目录的“清洗后审阅.csv”和“删除清单.csv”。X与y按行对应，audit保存来源，dataset.npz保存float32数组。'
        '\n\n本轮仅清洗抽取样本，未生成全量清洗数据、未划分训练集、未拟合scaler、未训练。正式训练入口仍为原版本，待样本审阅通过后统一接入NEW及新清洗规则。'
        '\n\n仅删除有明确依据的风量异常及完全重复记录，不按IQR删除、不插值、不推测修正值。直径700的组数不均衡保留待核查；ccm、power单位疑问保留原始数值。'
        '\n\n'+ '\n'.join(f"- {p}：清洗前 {r['sample_before']}，清洗后 {r['sample_after']}；几何组 {r['geometry_groups']}；删除ID {r['removed_ids']}。" for p,r in reports.items())+'\n',encoding='utf-8')
    return reports


if __name__ == '__main__':
    path = TRAINING_RESULTS_DIR / (datetime.now().strftime('%Y%m%d_%H%M%S')+'_data_review')
    export_review(path)
    print(f'审阅包：{path}',flush=True)
