"""Create a traceable XF dataset with cold-blow volume = process volume / 20."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from .candidate_data import load_candidate_dataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output must not exist; original datasets are retained')
    original = load_candidate_dataset(source, 'xf')
    output.mkdir(parents=True)
    files = {}
    for split, label in [('train', '训练集'), ('validation', '验证集'), ('test', '测试集')]:
        src = source / f'新风转轮_{label}.csv'
        dst = output / src.name
        before = sha(src)
        changed = 0
        with src.open(encoding='utf-8-sig', newline='') as inp, dst.open('w', encoding='utf-8-sig', newline='') as out:
            reader = csv.reader(inp)
            writer = csv.writer(out)
            header = next(reader)
            writer.writerow(header)
            process, cold = header.index('处理风量'), header.index('冷吹风量')
            for row in reader:
                corrected = Decimal(row[process]) / Decimal(20)
                changed += Decimal(row[cold]) != corrected
                row[cold] = format(corrected, 'f')
                writer.writerow(row)
        assert before == sha(src) == original.file_hashes[split]
        files[split] = {'source': str(src), 'source_sha256': before, 'output_sha256': sha(dst), 'changed_rows': changed}
    corrected = load_candidate_dataset(output, 'xf')
    for split, old in original.frames.items():
        new = corrected.frames[split]
        assert old.drop(columns='冷吹风量').equals(new.drop(columns='冷吹风量'))
        assert (abs(new['冷吹风量'] - new['处理风量'] / 20) < 1e-10).all()
        assert original.geometry_groups[split] == corrected.geometry_groups[split]
        inputs = new.loc[:, corrected.input_columns]
        labels = new.loc[:, corrected.output_columns]
        joined = inputs.join(labels)
        duplicate = inputs.duplicated(keep=False)
        conflicting = joined.loc[duplicate].groupby(list(corrected.input_columns)).nunique().max(axis=1).gt(1).sum()
        files[split].update(rows=len(new), geometry_groups=corrected.geometry_groups[split],
                            duplicate_input_rows=int(duplicate.sum()), conflicting_input_groups=int(conflicting))
    report = {'rule': '冷吹风量 = 处理风量 / 20', 'profile': 'xf', 'labels_unchanged': True,
              'ids_order_splits_unchanged': True, 'original_data_retained': True,
              'source_fingerprint': original.data_fingerprint, 'data_fingerprint': corrected.data_fingerprint, 'files': files}
    (output.parent / 'data_correction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
