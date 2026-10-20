"""风量版全量数据的8:1:1分层划分与分布审计，不执行训练。"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

STRATA = ['转轮厚度','转轮直径','再生温度(℃)']
SEED = 20260908


def assign_splits(x, seed=SEED):
    if not x.index.is_unique or x.duplicated().any():
        raise ValueError('存在重复输入或索引，拒绝跨集合泄漏')
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('输入存在非有限数值')
    strata=x[STRATA].astype(str).agg('|'.join,axis=1)
    train,hold=train_test_split(x.index,test_size=.2,random_state=seed,stratify=strata)
    validation,test=train_test_split(hold,test_size=.5,random_state=seed,stratify=strata.loc[hold])
    result=pd.Series('',index=x.index,name='split')
    for label,idx in [('train',train),('validation',validation),('test',test)]:
        result.loc[idx]=label
    assert result.ne('').all()
    return result


def assign_geometry_splits(x, seed=SEED):
    """完整留出厚度×直径组，仅按输入几何和行数选择划分。"""
    if x.duplicated().any() or not x.index.is_unique:
        raise ValueError('存在重复输入或索引')
    sizes=x.groupby(['转轮厚度','转轮直径'],sort=True).size()
    groups=list(sizes.index)
    rng=np.random.default_rng(seed)
    best=None
    # 每种厚度分配约10%尺寸组给验证和测试；其余训练。
    # 候选评分只用行数和直径分布，不读取y或模型表现。
    for _ in range(1000):
        mapping={}
        for thickness in sorted(x['转轮厚度'].unique()):
            g=[a for a in groups if a[0]==thickness]
            if len(g)<3:
                raise ValueError('每种厚度至少需要三个直径组')
            order=rng.permutation(len(g));n=max(1,round(len(g)*.1))
            for k,j in enumerate(order):
                mapping[g[j]]='validation' if k<n else 'test' if k<2*n else 'train'
        counts={label:sum(int(sizes[g]) for g in groups if mapping[g]==label) for label in ['train','validation','test']}
        ratio=sum(abs(counts[k]/len(x)-target) for k,target in [('train',.8),('validation',.1),('test',.1)])
        diam=np.array([g[1] for g in groups],dtype=float)
        distance=sum(abs(np.mean([g[1] for g in groups if mapping[g]==label])-diam.mean())/diam.std() for label in ['validation','test'])
        score=ratio*100+distance
        if best is None or score<best[0]:
            best=(score,mapping)
    return pd.Series([best[1][(t,d)] for t,d in zip(x['转轮厚度'],x['转轮直径'])],index=x.index,name='split')


def export_split(source, output, geometry=False):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    summaries={}
    for profile in ['xf','main']:
        print(f'{profile}：读取并验证全量数据',flush=True)
        root=source/profile
        hashes={}
        frames={}
        for name in ['X','y','来源']:
            b=(root/f'{name}.csv').read_bytes()
            hashes[name]=hashlib.sha256(b).hexdigest()
            frames[name]=pd.read_csv(io.BytesIO(b))
        x,y,orig=frames['X'],frames['y'],frames['来源']
        schema=json.loads((root/'schema.json').read_text())
        assert list(x)==schema['input_columns'] and list(y)==schema['output_columns']
        assert len(x)==len(y)==len(orig)==schema['clean_count']
        assert orig.ID.is_unique and orig['源SHA256'].eq(schema['sha256']).all()
        assert np.isfinite(y.to_numpy()).all()
        split=assign_geometry_splits(x) if geometry else assign_splits(x)
        dest=output/profile;dest.mkdir()
        assignment=orig.copy();assignment.insert(0,'全量行号',np.arange(len(x)));assignment['split']=split
        assignment.to_csv(dest/'划分索引.csv',index=False,encoding='utf-8-sig')
        stats=[];checks=[];all_data=pd.concat([x,y],axis=1)
        for label in ['train','validation','test']:
            idx=split.index[split.eq(label)]
            part=dest/label;part.mkdir()
            for name,frame in frames.items():
                frame.loc[idx].to_csv(part/f'{name}.csv',index=False,encoding='utf-8-sig')
            np.savez(part/'dataset.npz',X=x.loc[idx].to_numpy(np.float32),y=y.loc[idx].to_numpy(np.float32),source_id=orig.loc[idx,'ID'].to_numpy(np.int64))
            saved=np.load(part/'dataset.npz')
            np.testing.assert_array_equal(saved['source_id'],pd.read_csv(part/'来源.csv').ID)
            np.testing.assert_array_equal(saved['X'],pd.read_csv(part/'X.csv').to_numpy(np.float32))
            np.testing.assert_array_equal(saved['y'],pd.read_csv(part/'y.csv').to_numpy(np.float32))
            for col in all_data:
                values=all_data.loc[idx,col];full=all_data[col];std=full.std()
                ks=float(ks_2samp(values,full,method='asymp').statistic)
                smd=float(abs(values.mean()-full.mean())/std) if std else 0.
                stats.append({'集合':label,'字段':col,'数量':len(values),'最小值':values.min(),'P25':values.quantile(.25),'中位数':values.median(),'P75':values.quantile(.75),'最大值':values.max(),'均值':values.mean(),'标准差':values.std(),'KS距离_对全量':ks,'标准化均值差_对全量':smd})
                checks.append(ks<.03 and smd<.05)
        pd.DataFrame(stats).to_csv(dest/'分布对比.csv',index=False,encoding='utf-8-sig')
        groups=x[STRATA].copy();groups['split']=split
        group_counts=groups.groupby(STRATA+['split']).size().unstack(fill_value=0)
        if geometry:
            assert group_counts.gt(0).sum(axis=1).eq(1).all()
            geo=x[['转轮厚度','转轮直径']].assign(split=split)
            assert geo.groupby(['转轮厚度','转轮直径']).split.nunique().eq(1).all()
            geo.groupby(['split','转轮厚度','转轮直径']).size().rename('样本数').to_csv(dest/'尺寸组清单.csv',encoding='utf-8-sig')
        else:
            assert (group_counts>0).all().all()
        group_counts.to_csv(dest/'分层数量.csv',encoding='utf-8-sig')
        # 三集合来源无交集且并集覆盖全量；划分只由X及固定种子决定。
        ids=[set(orig.loc[split.eq(k),'ID']) for k in ['train','validation','test']]
        assert sum(map(len,ids))==len(set.union(*ids))==len(x)
        report={'profile':profile,'seed':SEED,'strata':STRATA[:2] if geometry else STRATA,'counts':split.value_counts().to_dict(),'source_file_hashes':hashes,'source_schema':schema,'max_KS':max(r['KS距离_对全量'] for r in stats),'max_standardized_mean_difference':max(r['标准化均值差_对全量'] for r in stats),'distribution_check_passed':all(checks),'thresholds':{'KS':.03,'standardized_mean_difference':.05},'scaled':False,'scope':'已知尺寸域内分层随机划分，同一几何组分布于三个集合，不用于证明新尺寸外推能力'}
        if geometry:
            report['scope']='厚度×直径组合完全隔离；同一直径可搭配不同厚度跨集合。不等于完全未见直径或厚度的外推。'
            report['selection']='固定种子1000个候选，仅按输入几何均值与行数偏差选择，不使用y'
            report['geometry_disjoint']=True
        (dest/'split_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        summaries[profile]=report
        print(f"{profile}：{report['counts']}；最大KS={report['max_KS']:.5f}，最大标准化均值差={report['max_standardized_mean_difference']:.5f}，分布检查={all(checks)}",flush=True)
    lines=['# 风量数据集8:1:1划分','','固定种子20260908，按厚度×直径×再生温度分层。各输入输出分别检查边际分布；有限样本无法保证分布完全相同，也不等于联合分布严格相同。未按测试输出挑选随机种子。','', '三集合无来源ID交集，覆盖全部清洗后数据，输入无完全重复。原始数据未修改；未标准化、未训练。后续只能用训练集拟合scaler；本测试集应固定保留。正式训练/仿真接口尚需迁移。','']
    for p,r in summaries.items():lines.append(f"- {p}：{r['counts']}；最大KS {r['max_KS']:.6f}；最大标准化均值差 {r['max_standardized_mean_difference']:.6f}；分布检查通过：{r['distribution_check_passed']}。")
    if geometry:
        lines[0]='# 按尺寸组隔离的数据集划分'
        lines[2]='厚度×直径作为不可拆分单元，各厚度约10%的直径组分配给验证和测试。固定种子20260908；1000个候选只根据行数比例和直径均值选择，不使用输出或模型成绩。三个集合尺寸组合完全不相交，但不保证直径本身不相交。组不可拆分，所以行数比例为近似8:1:1。尺寸与相关风量分布必然有差异，分布阈值只用于诊断，不为满足阈值破坏组隔离。'
    (output/'划分说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries


if __name__=='__main__':
    from .config import TRAINING_RESULTS_DIR
    import argparse
    parser=argparse.ArgumentParser(description='数据集划分')
    parser.add_argument('--geometry',action='store_true',help='按厚度×直径完整隔离')
    args=parser.parse_args()
    export_split(TRAINING_RESULTS_DIR/'20260908_full_data_distribution',TRAINING_RESULTS_DIR/('20260908_split_geometry_811' if args.geometry else '20260908_split_811'),geometry=args.geometry)
