"""Authenticated RC3/compiler-only adapter and existing measured-row utilities."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch
from torch_geometric.data import Data
from GNN_branch import mlir_graph_v2 as compiler
from GNN_branch.action_point_relations import action_point_views
from GNN_branch.target_graph_v22 import load_registry
from LLM_branch.common import directive_domains as domains
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
from Preprocessing.create_jsonl import build_examples, application_sources, load_actions, validate_preprocessing_manifest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'salt-paper-rc3-compiler-tensors-v1'
NUMERIC = ('loop_depth','trip_count','compiler_proven_trip_count','tcad_trip_count',
           'lower_bound','upper_bound','step','operand_count','result_count','use_count',
           'memory_rank','operand_index','dimension')
CATEGORIES = tuple(sorted(compiler.NODE_KINDS))
RELATIONS = tuple(sorted(compiler.EDGE_KINDS))
EVIDENCE = ('proven','possible','not_established','not_applicable')
RESOURCE_ORDER = ('BRAM_18K','DSP','FF','LUT')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def candidate_vector(assignment, actions):
    result = torch.zeros(64,8)
    for label, action in actions.items():
        i = int(label[1:])-1
        if not 0 <= i < 64: raise ValueError('Action truncation forbidden')
        get = lambda name: assignment['AUTO{_'+name+'_'+label+'}']
        if action.kind == 'loop':
            result[i,0] = float(get('PIPE'))
            result[i,1] = math.log2(1+int(get('UNROLL')))/12
        else:
            result[i,2+('none','complete','block','cyclic').index(get('ARRAY_T'))] = 1
            result[i,6] = math.log2(1+int(get('ARRAY_F')))/12
            result[i,7] = int(get('ARRAY_D'))/8
    return result


def tensorize(graph, vocabulary):
    if graph.graph.get('semantic_revision') != 'rc3': raise ValueError('RC3 required')
    nodes = sorted(graph.nodes)
    index = {n:i for i,n in enumerate(nodes)}
    features = []
    for node in nodes:
        a = graph.nodes[node]
        if a.get('_target_overlay') or a['category'] not in CATEGORIES:
            raise ValueError('Only original compiler nodes allowed')
        row = [float(a['category']==c) for c in CATEGORIES]
        token = str(a.get('text','')) if a['category']=='operation' else ''
        token = token if token in vocabulary else '<unknown>'
        row += [float(token==v) for v in vocabulary]
        for name in NUMERIC:
            value = a.get(name)
            known = isinstance(value,(int,float)) and math.isfinite(value)
            row += [math.copysign(math.log2(1+abs(value)),value)/24 if known else 0., float(known)]
        features.append(row)
    ends, edges = [], []
    for u,v,k,a in sorted(graph.edges(keys=True,data=True), key=lambda e:(e[0],e[1],str(e[2]))):
        relation = a['relation_type']
        if relation not in RELATIONS: raise ValueError('Noncompiler relation')
        for reverse in (False,True):
            ends.append((index[v],index[u]) if reverse else (index[u],index[v]))
            edges.append([float(relation==r and reverse==rev) for rev in (False,True) for r in RELATIONS]+
                         [float(a.get('evidence_status')==e) for e in EVIDENCE])
    active = sorted((int(a['action_id'][1:])-1,index[n]) for n,a in graph.nodes(data=True) if a['category']=='action_point')
    if len({i for i,n in active}) != len(active) or any(i>=64 or i<0 for i,n in active):
        raise ValueError('Ambiguous or truncated action identities')
    return Data(x=torch.tensor(features,dtype=torch.float32),
        edge_index=torch.tensor(ends,dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edges,dtype=torch.float32),
        action_slots=torch.tensor([i for i,n in active]), action_nodes=torch.tensor([n for i,n in active]))


class PaperData:
    def __init__(self, out):
        self.split_path = ROOT/'mailohls_runs/mailohls_final_family_split_s123.json'
        self.split = json.loads(self.split_path.read_text())
        families = [set(self.split[n+'_families']) for n in ('train','val','test')]
        assert all(not families[i]&families[j] for i in range(3) for j in range(i))
        selected = set(self.split['train_kernels']+self.split['val_kernels'])
        release_path = ROOT/'docs/graph-release-v22-rc3.json'
        release = json.loads(release_path.read_text())
        archive = ROOT/release['archive']
        if sha(archive)!=release['archive_sha256']: raise ValueError('RC3 archive hash changed')
        extracted = out/'authenticated_graphs'; extracted.mkdir(exist_ok=True)
        self.graphs, self.graph_hashes = {}, {}
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith('canonical/') and m.name.endswith('.json')]
            for kernel in sorted(selected):
                match = [m for m in members if Path(m.name).name.startswith(kernel+'-')]
                if len(match)!=1: raise ValueError('No unique RC3 graph for '+kernel)
                member=match[0]; raw=tar.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest()!=release['files_sha256'][member.name]: raise ValueError('RC3 member hash changed')
                path=extracted/Path(member.name).name; path.write_bytes(raw)
                self.graphs[kernel]=compiler.read_graph(path); self.graph_hashes[kernel]=sha(path)
        vocabulary={'<unknown>',''}
        for kernel in self.split['train_kernels']:
            vocabulary.update(str(a.get('text','')) for _,a in self.graphs[kernel].nodes(data=True) if a['category']=='operation')
        self.schema={'schema':SCHEMA,'vocabulary':sorted(vocabulary),'node_categories':CATEGORIES,
            'numeric':NUMERIC,'relations':RELATIONS,'evidence':EVIDENCE,'reverse_edges':True,
            'source_edge_metadata':'retained verbatim in authenticated graph JSON',
            'input_allowlist':'compiler categories, operation text, numeric compiler facts; no target/candidate/QoR/budget fields'}
        self.tensors={k:tensorize(g,self.schema['vocabulary']) for k,g in self.graphs.items()}
        registry=load_registry(); device_map={'xczu7ev-ffvc1156-2-e':'zcu104','xcu200-fsgd2104-2-e':'u200'}
        # Existing builder authenticates source/APL preprocessing manifests;
        # selecting train+val prevents parsing locked-test measurements.
        preprocessing=validate_preprocessing_manifest(ROOT/'LLM_branch/Data/preprocessed_CSVS')
        dataset_manifest=json.loads((ROOT/'artifacts/llm/mailohls_sft.manifest.json').read_text())
        if dataset_manifest['preprocessing_manifest_sha256']!=sha(preprocessing) or dataset_manifest['output_sha256']!=self.split['dataset_sha256']:
            raise ValueError('Selected preprocessing/dataset/split identities differ')
        rows, manifests, _=build_examples(ROOT/'LLM_branch/Data/preprocessed_CSVS', selected_kernels=selected)
        self.actions={k:domains.tcad.load_source_actions(k, ROOT/'Data/ApplicationDataset') for k in selected}
        sources=application_sources(selected)
        for kernel, source in sources.items():
            provenance=self.graphs[kernel].graph['provenance']
            if sha(source)!=provenance['source_sha256']:
                raise ValueError('RC3 source identity differs from measured source: '+kernel)
            if sha(source.parent/'kernel_info.txt')!=provenance['metadata_sha256']:
                raise ValueError('RC3 absolute-action metadata identity changed: '+kernel)
        missing={k:any(not a.csv_column for a in load_actions(k,sources[k].read_text())) for k in selected}
        coverage=Counter(); self.groups=defaultdict(list); seen={}
        floor=float(self.split['effective_area_floor'])
        for row in rows:
            k=row['kernel_name']; coverage['input_rows']+=1
            if missing[k]: coverage['missing_measured_apl_column']+=1; continue
            device=row['device'].lower()
            if device not in device_map: coverage['unsupported_device']+=1; continue
            assignment={}
            for line in row['target'].splitlines():
                key,value=line.split('=',1); assignment[key.strip().upper()]=value.strip().lower()
            try: domains.tcad.validate_assignments(assignment,self.actions[k])
            except ValueError: coverage['illegal_complete_configuration']+=1; continue
            caps=[float(registry['profiles'][device_map[device]]['resources'][r]['value']) for r in RESOURCE_ORDER]
            resources=[float(row[c])*cap/100 for c,cap in zip(('bram_util_%','dsp_util_%','ff_util_%','lut_util_%'),caps)]
            key=(k,device,float(row['clock_period'])); design=fingerprint(assignment)
            target=[math.log2(row['latency']),math.log2(max(row['area'],floor))]+[math.log1p(x) for x in resources]
            identity=(*key,design)
            if identity in seen:
                if seen[identity]!=target: raise ValueError('Conflicting duplicate physical measurements')
                coverage['duplicate_physical_designs']+=1; continue
            seen[identity]=target
            record={'kernel':k,'family':family_id_from_kernel_name(k),'device':device,'clock':key[2],
                'split':'train' if k in self.split['train_kernels'] else 'val',
                'design_id':design,'target':target,'resources':resources,'capacities':caps,
                'candidate':candidate_vector(assignment,self.actions[k]),'assignment':assignment,
                'row_index':row['preprocessed_row']}
            self.groups[key].append(record); coverage['accepted_unique_designs']+=1
        self.coverage=dict(coverage)
        training=[r['target'] for rows in self.groups.values() for r in rows if r['split']=='train']
        raw=torch.tensor(training,dtype=torch.float64)
        self.mean=raw.mean(0).float(); self.std=raw.std(0,unbiased=False).clamp_min(1e-6).float()
        self.adp_std=(raw[:,0]+raw[:,1]).std(unbiased=False).clamp_min(1e-6).float()
        self.identity={'split_sha256':sha(self.split_path),'release_manifest_sha256':sha(release_path),
            'archive_sha256':sha(archive),'graph_sha256':self.graph_hashes,'feature_schema':self.schema,
            'measurement_manifests':manifests,'normalization_mean':self.mean.tolist(),'normalization_std':self.std.tolist(),
            'coverage':self.coverage,'locked_test_read':False,'area_floor':floor,'adp_log2_std':float(self.adp_std),
            'units':{'latency':'ms','area':'arithmetic mean of resource utilization percentages',
                     'adp':'ms * area; equivalent within clock pools to cycles * area ordering/regret',
                     'resources':'physical BRAM18K,DSP,FF,LUT from measured percentages and authenticated device capacities'},
            'measurement_policy':'successfully preprocessed complete HLS designs; failures/unknown excluded, never labelled infeasible'}

    @staticmethod
    def context(record, device):
        return torch.tensor([[math.log2(c)/24 for c in record['capacities']]+[math.log2(record['clock']/10)]],device=device)
