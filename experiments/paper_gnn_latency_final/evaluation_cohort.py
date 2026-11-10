"""User-requested four-kernel evaluation amendment: radix L6 is unobserved/ignored."""
import math
from collections import Counter
import json
import torch
from Preprocessing.create_jsonl import build_examples
from LLM_branch.common import directive_domains as domains
from LLM_branch.common.mailohls_contract import family_id_from_kernel_name
from GNN_branch.target_graph_v22 import load_registry
from experiments.paper_gnn.data import candidate_vector,fingerprint,RESOURCE_ORDER


def include_radix_ignoring_l6(data,cohort):
    kernel='machsuite-sort-radix'
    assert kernel in data.split['val_kernels'] and kernel not in data.split['train_kernels']
    observed={k:v for k,v in data.actions[kernel].items() if k!='L6'}
    rows,manifests,_=build_examples(cohort,selected_kernels={kernel});counts=Counter();registry=load_registry();devices={'xczu7ev-ffvc1156-2-e':'zcu104','xcu200-fsgd2104-2-e':'u200'}
    seen=set()
    for row in rows:
        assignment={}
        for line in row['target'].splitlines():
            lhs,value=line.split('=',1);lhs=lhs.strip().upper()
            if lhs.endswith('_L6}'):continue
            assignment[lhs]=value.strip().lower()
        counts['input_rows']+=1
        try:domains.tcad.validate_assignments(assignment,observed)
        except ValueError:counts['illegal_observed_assignment']+=1;continue
        device=row['device'].lower()
        if device not in devices:counts['unsupported_device']+=1;continue
        caps=[float(registry['profiles'][devices[device]]['resources'][n]['value']) for n in RESOURCE_ORDER]
        percentages=[float(row[n]) for n in ('bram_util_%','dsp_util_%','ff_util_%','lut_util_%')];resources=[p*c/100 for p,c in zip(percentages,caps)]
        key=(kernel,device,float(row['clock_period']));design=fingerprint(assignment)
        if (*key,design) in seen:raise ValueError('Duplicate observed configuration after excluding L6; cannot create independent observations')
        seen.add((*key,design))
        target=[math.log2(row['latency']),math.log2(max(row['area'],data.identity['area_floor']))]+[math.log1p(v) for v in resources]
        data.groups[key].append(dict(kernel=kernel,family=family_id_from_kernel_name(kernel),device=device,clock=key[2],split='val',design_id=design,target=target,resources=resources,capacities=caps,candidate=candidate_vector(assignment,observed),assignment=assignment,row_index=row['preprocessed_row'],resource_percentages=percentages,ignored_actions=['L6'],candidate_observation_status='all registered actions measured except L6, explicitly ignored by user'))
        counts['accepted_observed_configurations']+=1
    return {'schema':'salt-final-latency-user-evaluation-amendment-v1','request':'4 validation kernels: sort-radix (ignoring L6), viterbi, rodinia_pathfinder_0_baseline_0, rodinia_pathfinder_4_doublebuffer_0','validation_kernels':data.split['val_kernels'],'ignored_action':{kernel:['L6']},'counts':dict(counts),'measurement_manifests':manifests,'training_cohort_changed':False,'original_training_checkpoint_selection_cohort_changed':False,'amendment_after_A_training':True,'qualification':'Radix comparisons concern configurations observed at ten actions. L6 is not assigned, predicted, guessed or supervised; its absent candidate coordinates are masked zeros. Unknown L6 differences cannot be inferred. This is the user-requested projected evaluation, not an authentication of the missing raw directive.'}
