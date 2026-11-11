"""Inspect the retained GEMV graph/action identities using only Python's stdlib."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    graph = ROOT / 'paper_examples/gemv/gemv_mlir.gexf'
    manifest = json.loads((ROOT / 'paper_figures/gemv_mlir_reduced.json').read_text())
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    if digest != manifest['source_sha256']:
        raise ValueError('Graph identity differs from the retained figure manifest')
    metadata = ROOT / 'evaluation_set/gemv/code_to_memory_outputs/meta/kernel_info.txt'
    actions = {row.split(',')[0]: row.split(',')[1]
               for row in metadata.read_text().splitlines()[1:] if row.strip()}
    if set(actions) != set(manifest['all_actions']):
        raise ValueError('Source action set differs from the retained graph manifest')
    for action, record in manifest['all_actions'].items():
        if actions[action] != record['kind']:
            raise ValueError('Action kind mismatch: ' + action)
    tree = ET.parse(graph)
    nodes = tree.findall('.//{*}nodes/{*}node')
    edges = tree.findall('.//{*}edges/{*}edge')
    print('Retained production graph SHA256:', digest)
    print('Full graph:', len(nodes), 'nodes,', len(edges), 'edges')
    for action in manifest['selected_actions']:
        print(action, actions[action], '-> absolute memory slot', int(action[1:]) - 1)
    print('Inspection only; no learned embeddings, decoding or hardware validation performed.')


if __name__ == '__main__':
    main()
