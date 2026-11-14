"""Narrow, explicit evaluation provenance extensions; no graph hash rewriting."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys

GENERATOR_SHA='5e4f93c89ce4c275c04ff6ed4dd81a17858647a55b0abec0846b8da240da3b05'
ROSETTA_SYMBOL='_Z8zculling7ap_uintILi16EEP14CandidatePixelS0_P5Pixel@static@_ZZ8zculling7ap_uintILi16EEP14CandidatePixelS0_P5PixelE8z_buffer'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_generator(path,name):
    if sha(path)!=GENERATOR_SHA:raise RuntimeError('Pinned generator bytes changed')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def install_graph_hooks(mod,kernel,audit):
    if kernel=='Kalman':
        original=mod.build_cgeist_command
        def command(**kwargs):
            cmd=original(**kwargs)
            if kwargs['kernel']!='krnl_KALMAN':raise RuntimeError('Kalman compiler hook used for another function')
            cmd.insert(cmd.index('-o'),'--mailohls-preserve-action-storage')
            return cmd
        mod.build_cgeist_command=command
    elif kernel=='rosetta-3d-rendering':
        original=mod.MLIRGraphBuilder._array_argument_node
        def resolve(self,action):
            if action.action_id!='L1' or action.variable!='z_buffer':return original(self,action)
            if action.function!='zculling' or tuple(action.array_dimensions)!=(256,256):
                raise RuntimeError('Static global source declaration changed')
            definitions=[]
            for op in self.module.body.operations:
                if op.operation.name!='memref.global':continue
                if str(op.attributes['sym_name']).strip('"')==ROSETTA_SYMBOL:
                    definitions.append(mod.operation_first_line(op))
            if len(definitions)!=1 or 'memref<256x256x1xi64>' not in definitions[0]:
                raise RuntimeError('Exact compiler global definition missing or ambiguous')
            candidates=[]
            for record in self.operation_records:
                if record.op_name!='memref.get_global':continue
                line=mod.operation_first_line(record.operation)
                m=re.search(r'memref.get_global @"([^"]+)"',line)
                if not m or m.group(1)!=ROSETTA_SYMBOL:continue
                if not record.function_name.startswith('_Z8zculling'):continue
                for key in record.result_keys:
                    node=self.value_nodes[key]
                    if self._node_memory_shape(node)!=(256,256,1):continue
                    lines=self._root_access_source_lines(node,action.source_file)
                    if not set(action.source_use_lines).intersection(lines):continue
                    accesses=[x for x in self.memory_accesses if x.root_node==node]
                    modes={x.mode for x in accesses}
                    if 'read' not in modes or 'write' not in modes:continue
                    candidates.append((record,node,sorted(lines),accesses))
            if len(candidates)!=1:raise RuntimeError('Global action must resolve to one source-accessed compiler root')
            record,node,lines,accesses=candidates[0]
            audit.append({'kernel':kernel,'action':'L1','variable':'z_buffer','resolution':'global_symbol_and_access',
                'source_file':action.source_file,'source_line':action.source_line,'source_column':action.source_column,
                'symbol':ROSETTA_SYMBOL,'global_definition':definitions[0],'native_root_value_node':node,
                'native_get_global_function':record.function_name,'native_get_global_ordinal':record.function_ordinal,
                'source_use_lines':sorted(action.source_use_lines),'compiler_access_source_lines':lines,
                'compiler_accesses':[{'function_id':x.function_id,'operation_ordinal':x.op_ordinal,'mode':x.mode} for x in accesses],
                'mlir_sha256':hashlib.sha256(self.mlir_text.encode()).hexdigest()})
            print('[ROSETTA VERIFIED GLOBAL]',ROSETTA_SYMBOL,'access lines',lines,flush=True)
            return record.function_id,node,record.block_id,'global_symbol_and_access'
        mod.MLIRGraphBuilder._array_argument_node=resolve


def install_gmt_scope_extension(verified_graphs,generated_directory):
    """Accept only receipt-authenticated Rosetta L1 global provenance.

    GNN tensorizer/features, all scopes/evidence/routing and GMT math remain
    unchanged. The two old GMT functions' source_location-only checks are
    extended with a predicate requiring the exact global/source/access proof.
    Graph action_resolution stays global_symbol_and_access throughout.
    """
    from experiments.historical_mlir_gmt_gemv import historical_gmt_slots as slots
    originals={name:getattr(slots,name) for name in ('cohort_action_mapping','make_evidence')}
    registered={}
    def allowed(graph,attrs):
        if attrs.get('action_resolution')=='source_location':return True
        item=registered.get(id(graph))
        return (item is not None and item[0] is graph and attrs is graph.nodes[item[1]] and
                attrs.get('action_id')=='L1' and attrs.get('action_resolution')=='global_symbol_and_access')
    def authenticate(graph,pack):
        unusual=[(n,a) for n,a in graph.nodes(data=True) if a.get('action_resolution') and a['action_resolution']!='source_location']
        if not unusual:return
        if len(unusual)!=1:raise RuntimeError('Only one explicitly proved static-global anchor is allowed')
        digest=pack['source_gexf_sha256'];record=verified_graphs.get(digest)
        if record is None:raise RuntimeError('Graph SHA not in evaluation compatibility allowlist')
        proof=record['native_binding'];gp=pack['graph_provenance']
        for key in ['source_sha256','action_sha256','generator_sha256','mlir_sha256','cgeist_sha256']:
            if gp[key]!=record[key]:raise RuntimeError('Evaluation provenance mismatch: '+key)
        if gp['generator_sha256']!=GENERATOR_SHA or proof['mlir_sha256']!=gp['mlir_sha256']:
            raise RuntimeError('MLIR/global proof does not match unchanged generator output')
        n,attrs=unusual[0]
        if not (int(attrs['type'])==104 and attrs['action_id']=='L1' and attrs['array_var']=='z_buffer' and
                attrs['action_resolution']=='global_symbol_and_access' and proof['symbol']==ROSETTA_SYMBOL and
                int(attrs['action_source_line'])==proof['source_line'] and int(attrs['action_source_column'])==proof['source_column']):
            raise RuntimeError('Global action identity/source location mismatch')
        roots=[v for _,v,d in graph.out_edges(n,data=True) if d.get('role')=='array_root' and int(d['flow'])==7]
        accesses=[(v,d) for _,v,d in graph.out_edges(n,data=True) if str(d.get('role','')).startswith(('array_read','array_write')) and int(d['flow'])==7]
        if len(roots)!=1 or ROSETTA_SYMBOL not in graph.nodes[roots[0]]['full_text']:
            raise RuntimeError('Global action has no exact native get_global root')
        if not accesses or not any(d['role']=='array_read' for _,d in accesses) or not any(d['role']=='array_write' for _,d in accesses):
            raise RuntimeError('Global action lacks genuine read/write scope edges')
        if not set(proof['source_use_lines']).intersection(proof['compiler_access_source_lines']):
            raise RuntimeError('Global access lines do not intersect exact source uses')
        registered[id(graph)]=(graph,n)
    generated_directory=Path(generated_directory);generated_directory.mkdir(parents=True,exist_ok=True)
    source_records={}
    for name,fn in originals.items():
        text=inspect.getsource(fn)
        old="if not action or attrs.get('action_resolution')!='source_location':" if name=='cohort_action_mapping' else "if str(a.get('action_id'))!=m['action'] or a.get('action_resolution')!='source_location':"
        new="if not action or not _compat_allowed(graph,attrs):" if name=='cohort_action_mapping' else "if str(a.get('action_id'))!=m['action'] or not _compat_allowed(graph,a):"
        if text.count(old)!=1:raise RuntimeError('Original GMT scope checker differs; refusing an unchecked runtime patch')
        replacement=text.replace(old,new)
        dest=generated_directory/(name+'_eval_global_v2.py')
        if dest.exists() and dest.read_text()!=replacement:raise RuntimeError('Generated compatibility code changed')
        dest.write_text(replacement)
        namespace=dict(fn.__globals__);namespace['_compat_allowed']=allowed
        exec(compile(replacement,str(dest),'exec'),namespace)
        revised=namespace[name]
        if name=='cohort_action_mapping':
            def wrapper(graph,pack,_revised=revised):
                authenticate(graph,pack)
                return _revised(graph,pack)
            setattr(slots,name,wrapper)
        else:setattr(slots,name,revised)
        source_records[name]={'original_function_sha256':hashlib.sha256(text.encode()).hexdigest(),
                              'generated_function_sha256':sha(dest),'generated_file':str(dest)}
    return {'schema':'mailohls-eval-global-scope-provenance-extension-v2',
            'original_module_sha256':sha(slots.__file__),'hook_sha256':sha(__file__),
            'functions':source_records,'graph_sha256_allowlist':sorted(verified_graphs),
            'graph_attributes_rewritten':False,'gnn_tensorizer_changed':False,'gmt_weights_or_math_changed':False}
