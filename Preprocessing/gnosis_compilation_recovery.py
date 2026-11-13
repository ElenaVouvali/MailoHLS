"""Versioned compiler-input correspondences, separate from the frozen v1 receipt.

Only the four enumerated Vitis-2021.1 comparisons can promote canonical inputs.
KNN remains a different source version and has an explicit measured-source join.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from Preprocessing import gnosis_recovery as recovery

PROOF = Path('Data/provenance/gnosis_5_compilation_20261113')
REGISTRY = PROOF/'correspondence.json'
CONTRACT = 'canonical-csv-apl-plus-vitis2021.1-source-correspondence-v2'
ORIGINAL_CONTRACT = 'gnosis-measured-source-8888db17-sqlite-states-v1'
SYMBOLS = {'spcl_example_03': ('_Z9Stencil2DPKfPf', 'Stencil2D'),
           'spcl_example_05': ('_Z20MatrixMultiplicationPKdS0_Pd', 'MatrixMultiplication')}
DILATE = {'rodinia_dilate_2_pipeline_0', 'rodinia_dilate_3_pipeline_0'}
KNN = 'rodinia-knn-5-coalescing'


def normalize_ir(text, symbol=None):
    """Remove file identity only; retain all IR semantics and HLS annotations."""
    text = '\n'.join(line for line in text.splitlines()
                     if not line.startswith(('; ModuleID = ', 'source_filename = ')))+'\n'
    if symbol:
        old, new = symbol
        if len(re.findall(r'^define .*?@'+re.escape(old)+r'\(', text, re.M)) != 1:
            raise ValueError('Original unique top binding absent')
        text = text.replace('@'+old+'(', '@'+new+'(')
    return text


def preprocessed_body(kernel, text):
    if kernel in DILATE:
        # Exact source starts this block after includes; no body is rewritten.
        return text[text.rindex('extern "C" {'):]
    if kernel in SYMBOLS:
        text = text.replace('extern "C"', '')
        return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|\d+|\S', text)
    raise ValueError('No authenticated canonical preprocessing rule for '+kernel)


def read_text(root, path):
    with gzip.open(root/path, 'rt') as handle:
        return handle.read()


def validate(kernel, root=recovery.ROOT):
    path = root/REGISTRY
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt['schema'] != 'gnosis-five-source-correspondence-v1':
        raise ValueError('Unknown compiler correspondence schema')
    entry = receipt['kernels'].get(kernel)
    if entry is None:
        return None
    if recovery.sha(root/recovery.REGISTRY) != receipt['parent_registry_sha256']:
        raise ValueError('Original action recovery receipt changed')
    for name, digest in {**receipt['shared_files'], **entry['files']}.items():
        if recovery.sha(root/name) != digest:
            raise ValueError('Compiler correspondence input changed: '+name)
    pinned = {r['path']:r['blob'] for r in json.loads((root/recovery.PROOF/'upstream_blobs.json').read_text())}
    for name, record in entry['original_blobs'].items():
        raw = (root/name).read_bytes()
        blob = hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
        if blob != pinned[record['upstream_path']] or blob != record['git_blob']:
            raise ValueError('Measured source differs from the pinned upstream bundle')
    with gzip.open(root/entry['graph']['gexf'], 'rb') as stream:
        graph = ET.parse(stream).getroot().find('{*}graph')
    metadata = json.loads(graph.attrib['name'].split('mailohls-meta-v1:', 1)[1])
    source, = (root/entry['original_bundle']).glob('*.cpp')
    source_actions = set(re.findall(r'\b(L[1-9][0-9]*)\s*:', source.read_text()))
    attribute_ids = {a.attrib['id'] for a in graph.findall('{*}attributes/{*}attribute')
                     if a.attrib.get('title') == 'action_id'}
    graph_actions = {a.attrib['value'] for a in graph.findall('{*}nodes/{*}node/{*}attvalues/{*}attvalue')
                     if a.attrib['for'] in attribute_ids and a.attrib['value']}
    if (metadata['source_sha256'] != recovery.sha(source)
            or metadata['source_sha256'] != entry['graph']['source_sha256']
            or metadata['action_resolutions'] != {'source_location':len(source_actions)}
            or graph_actions != source_actions):
        raise ValueError('Compiler graph source/action correspondence differs')
    if entry['canonical_eligible']:
        if kernel not in DILATE | set(SYMBOLS):
            raise ValueError('Kernel has no authenticated canonical correspondence')
        llvm = {kind:read_text(root, entry['evidence'][kind]['llvm'])
                for kind in ('original', 'canonical')}
        if normalize_ir(llvm['original'], SYMBOLS.get(kernel)) != normalize_ir(llvm['canonical']):
            raise ValueError('HLS compiler modules are different')
        pp = {kind:preprocessed_body(kernel, read_text(root, entry['evidence'][kind]['preprocessed']))
              for kind in ('original', 'canonical')}
        if pp['original'] != pp['canonical']:
            raise ValueError('Effective source bodies are different')
        # No overload or different top selection is allowed by the correspondence.
        original_top = (root/entry['original_bundle']/'kernel_info.txt').read_text().splitlines()[0]
        canonical_top = (root/'Data/ApplicationDataset'/kernel/'kernel_info.txt').read_text().splitlines()[0]
        if original_top != entry['top'] or canonical_top != entry['top']:
            raise ValueError('Top-function binding differs')
    return entry


def canonical_entry(entry, root=recovery.ROOT):
    proof = validate(entry['kernel'], root)
    if proof is None or not proof['canonical_eligible']:
        return entry
    return dict(entry, canonical_eligible=True, contract=CONTRACT,
                compilation_correspondence={'registry':str(REGISTRY),
                    'registry_sha256':recovery.sha(root/REGISTRY),
                    'historical_per_measurement_flags_available':False,
                    'basis':proof['basis']})


def complete_original_rows(kernel, root=recovery.ROOT):
    """Original-source rows only; cannot be mistaken for the canonical KNN input."""
    split = json.loads((root/'mailohls_runs/mailohls_final_family_split_s123.json').read_text())
    if kernel in split['test_kernels']:
        raise ValueError('Locked-test rows cannot be loaded by the recovery join')
    proof = validate(kernel, root)
    if kernel != KNN or proof is None or not proof['measured_source_eligible']:
        raise ValueError('No qualified separate measured-source dataset for '+kernel)
    entry = recovery.validated_entry(kernel, root)
    for row in recovery._joined_rows(entry, root, ORIGINAL_CONTRACT):
        yield dict(row, source_version=proof['source_version'],
                   source_bundle=proof['original_bundle'], compiler_graph=proof['graph'])


def verify(root=recovery.ROOT):
    receipt = json.loads((root/REGISTRY).read_text())
    canonical_rows = original_rows = 0
    for kernel in receipt['kernels']:
        entry = validate(kernel, root)
        if entry['canonical_eligible']:
            canonical_rows += sum(1 for _ in recovery.complete_rows(kernel, root))
        if kernel == KNN:
            original_rows += sum(1 for _ in complete_original_rows(kernel, root))
    eligible = [e for e in receipt['kernels'].values() if e['canonical_eligible']]
    result = dict(status='PASS', new_canonical_kernels=len(eligible),
                  new_canonical_action_histories=sum(len(e['recovered_actions']) for e in eligible),
                  canonical_joined_rows=canonical_rows, separate_measured_source_kernels=1,
                  separate_measured_source_rows=original_rows, imputed_states=0)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', required=True)
    parser.parse_args()
    verify()
