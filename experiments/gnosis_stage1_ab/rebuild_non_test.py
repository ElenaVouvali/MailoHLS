"""Rebuild target SFT data from RESTORED canonical GNOSIS, with no test rows accessed.

Creates a source-separated train/validation-only GNOSIS view. It never opens
CSV measurements for the locked-test kernel or canonical-incomplete kernels.
The original production preprocessing and JSONL serialiser are reused unchanged.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def sha(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--canonical-manifest',type=Path,default=Path('Data/manifest.json'))
    args=ap.parse_args();z=args.out
    if z.exists():raise FileExistsError('Immutable output exists: '+str(z))
    man=json.loads(args.canonical_manifest.read_text());assert man['schema']=='salt-hls-canonical-data-v1'
    selected={x['kernel']:x for x in man['kernels'] if x['split'] in ('train','val') and x['complete_source_configuration_eligible']}
    if not selected or not any(x['split']=='val' for x in selected.values()):raise ValueError('No valid train/val canonical cohort')
    source=z/'raw';source.mkdir(parents=True);read_hashes={}
    for name,x in sorted(selected.items()):
        item=x['files']['measurements'];path=Path(item['path'])
        if not path.is_file():raise FileNotFoundError(path)
        got=sha(path)
        if got!=item['sha256']:raise ValueError(f'Canonical file mismatch {name}: {got} vs {item["sha256"]}')
        (source/(name+'.csv')).symlink_to(path.resolve())
        read_hashes[name]=got
    # No locked-test CSV is even present in this staged input directory.
    preprocess=z/'preprocessed';dataset=z/'mailohls_sft.jsonl'
    subprocess.run([sys.executable,'-m','Preprocessing.data_preprocess','--mode','llm',
                    '--input-dir',str(source),'--output-dir',str(preprocess)],check=True)
    subprocess.run([sys.executable,'-m','Preprocessing.create_jsonl',
                    '--preprocessed-dir',str(preprocess),'--output',str(dataset)],check=True)
    train=[];val=[];count={}
    for idx,line in enumerate(dataset.open()):
        # This file only includes 37 verified non-test kernels.
        row=json.loads(line);name=row['kernel_name'];fold=selected[name]['split'];
        (train if fold=='train' else val).append(idx)
        count[name]=count.get(name,0)+1
    assert train and val and len(train)+len(val)==sum(count.values())
    split={'schema':'gnosis-canonical-trainval-only-v1','train_jsonl_idx':train,'val_jsonl_idx':val,
           'test_jsonl_idx':[], 'locked_test_measurements_read':False,
           'train_kernels':sorted(x for x in count if selected[x]['split']=='train'),
           'val_kernels':sorted(x for x in count if selected[x]['split']=='val'),
           'test_kernels_not_present':[x['kernel'] for x in man['kernels'] if x['split']=='test'],
           'dataset_sha256':sha(dataset),'canonical_manifest_sha256':sha(args.canonical_manifest)}
    (z/'split_train_val.json').write_text(json.dumps(split,indent=2)+'\n')
    result={'schema':'gnosis-canonical-source-build-v1','included_kernels':len(selected),
            'train_rows':len(train),'val_rows':len(val),'test_rows':0,
            'raw_measurement_sha256':read_hashes,'dataset_sha256':sha(dataset),
            'sft_manifest_sha256':sha(dataset.with_suffix('.manifest.json')),
            'split_sha256':sha(z/'split_train_val.json')}
    (z/'build_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
