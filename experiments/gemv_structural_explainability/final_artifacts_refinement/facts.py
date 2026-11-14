"""Explicit compiler/source facts with separate knownness; no QoR features."""
import json
import math
import re

NUMERIC=('trip_count','loop_depth','memory_rank',*[f'extent_{i}' for i in range(8)],
         'loop_lower','loop_upper','loop_step','operand_count','result_count','use_count',
         'map_inputs','map_outputs','map_dim_references','map_symbol_references',
         'map_adds','map_muls','map_divs','operand_index','source_rank','scalarized')
FLAGS=('read','write','readwrite','ssa_recurrence','memory_RAW','memory_WAR','memory_WAW',
       'loop_parent_child','multiply','add','load','store','integer','float','index','memref')
FACT_NAMES=tuple(name+suffix for name in NUMERIC+FLAGS for suffix in ('_value','_known'))


def encode_facts(record,graph):
    payload=record['payload'];values={}
    node=graph.nodes[record['node']] if record['kind']=='node' else graph.nodes[record['source']]
    for name in ('trip_count','loop_depth','loop_lower','loop_upper','loop_step','operand_count','result_count','use_count'):
        if isinstance(node.get(name),(int,float)):values[name]=float(node[name])
    shape=node.get('memory_shape')
    if shape is not None:
        if len(shape)>8:raise ValueError('Memref rank exceeds explicit fact schema; no truncation')
        values['memory_rank']=len(shape)
        for i,extent in enumerate(shape):
            if isinstance(extent,int) and extent>=0:values[f'extent_{i}']=extent
    if record['kind']=='node':
        for n,a in graph.nodes(data=True):
            if a.get('category')!='action_point' or a.get('action_kind')!='array':continue
            targets=[v for u,v,k,e in graph.out_edges(n,keys=True,data=True) if e['relation_type']=='targets' and e.get('evidence_status')=='proven']
            if record['node'] in targets and isinstance(a.get('source_array_dimensions'),list):
                values['source_rank']=len(a['source_array_dimensions'])
                if shape is not None:values['scalarized']=int(len(shape)!=len(a['source_array_dimensions']))
    if isinstance(payload.get('operand_index'),(int,float)):values['operand_index']=payload['operand_index']
    access=payload.get('access')
    if access in ('read','write','readwrite'):
        for role in ('read','write','readwrite'):values[role]=int(role==access)
    if record['type']=='loop_carried':values['ssa_recurrence']=1
    if record['type']=='loop_child':values['loop_parent_child']=1
    if payload.get('dependence_kind') in ('RAW','WAR','WAW'):
        for kind in ('RAW','WAR','WAW'):values['memory_'+kind]=int(kind==payload['dependence_kind'])
    text=str(node.get('text',''))
    if node.get('category')=='operation':
        for flag,token in [('multiply','mul'),('add','add'),('load','load'),('store','store')]:
            values[flag]=int(token in text.split('.')[-1])
    value_type=str(node.get('value_type',''))
    if value_type:
        values.update(integer=int(bool(re.fullmatch('[su]?i[0-9]+',value_type))),
                      float=int(bool(re.fullmatch('(bf|f)[0-9]+',value_type))),
                      index=int(value_type=='index'),memref=int('memref<' in value_type))
    access_indices=payload.get('access_indices') or {}
    mapping=access_indices.get('affine_map')
    if not mapping:
        attrs=node.get('mlir_attrs') or {}
        if isinstance(attrs,str):attrs=json.loads(attrs)
        mapping=attrs.get('map')
    if mapping:
        printed=str(mapping)
        match=re.search(r'<\((.*?)\)(?:\[(.*?)\])?\s*->\s*\((.*?)\)>',printed)
        if match:
            lhs,symbols,rhs=match.groups()
            values.update(map_inputs=len(lhs.split(',')) if lhs.strip() else 0,
                map_outputs=len(rhs.split(',')) if rhs.strip() else 0,
                map_dim_references=len(re.findall(r'\bd\d+\b',rhs)),
                map_symbol_references=len(re.findall(r'\bs\d+\b',rhs)),
                map_adds=rhs.count('+')+rhs.count('-'),map_muls=rhs.count('*'),
                map_divs=len(re.findall('floordiv|ceildiv|mod',rhs)))
    result=[]
    for name in NUMERIC+FLAGS:
        value=values.get(name)
        known=isinstance(value,(int,float)) and math.isfinite(value)
        result += [math.copysign(math.log2(1+abs(value)),value)/16 if known else 0.,float(known)]
    return result
