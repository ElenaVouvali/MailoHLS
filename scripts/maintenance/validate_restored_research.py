"""Validate sealed original paths and the graph/table readers, without training."""
import argparse
import csv
import json
from pathlib import Path
import tarfile

import networkx as nx

from scripts.artifacts import MANIFEST, sha256, verify


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    counts = dict(files=0, csv_files=0, csv_rows=0, json_files=0, gexf_files=0, graph_nodes=0, graph_edges=0)
    summaries=[]
    for record in json.loads(MANIFEST.read_text())['artifacts']:
        if not record.get('restore_roots'):
            continue
        archive=args.root/record['path']
        verify(archive,record)
        with tarfile.open(archive) as tar:
            manifest=json.load(tar.extractfile('MAILOHLS_ARTIFACT.json'))
        for row in manifest['files']:
            path=args.root/row['path']
            verify(path,row)
            assert path.stat().st_mode & 0o111 == row['mode'] & 0o111, path
            counts['files']+=1
            if path.suffix=='.csv':
                with path.open(newline='') as stream:
                    reader=csv.DictReader(stream)
                    assert reader.fieldnames, path
                    counts['csv_rows']+=sum(1 for _ in reader)
                counts['csv_files']+=1
            elif path.suffix=='.json':
                json.loads(path.read_text())
                counts['json_files']+=1
            elif path.suffix=='.gexf':
                graph=nx.read_gexf(path)
                counts['gexf_files']+=1
                counts['graph_nodes']+=graph.number_of_nodes()
                counts['graph_edges']+=graph.number_of_edges()
        summaries.append(dict(id=record['id'],sha256=sha256(archive),files=len(manifest['files'])))
    result=dict(schema='mailohls-restored-consumers-v1',counts=counts,artifacts=summaries,limitations='Original graph/table reader validation; no model-weight loading, HLS execution or new QoR measurement.')
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    main()
