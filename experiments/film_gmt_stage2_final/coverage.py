"""Deterministic exposure coverage, without accessing locked-test outcomes."""
import collections
import hashlib
import json


def physical(row):
    return (row['kernel_name'],row['device'],float(row.get('selected_clock_period',row['clock_period'])))


def paired_groups(pairs):
    groups=collections.defaultdict(list)
    for pair in pairs:groups[(pair['kind'],pair['group_id'])].append(pair)
    for key,values in groups.items():
        if len(values)!=2:raise ValueError('A preference group must retain both target contexts: '+str(key))
        if len({tuple(physical(v['row']))+(v['row'].get('resource_budget_id'),) for v in values})!=2:
            raise ValueError('Preference group has no distinct target contexts')
    return groups


def exposure_report(data,schedule,validation,steps,grad_accum):
    contexts,selected,pairs=schedule
    visits=collections.Counter(selected[i%len(selected)] for i in range(steps*grad_accum))
    observed={physical(contexts[k][0]['row']) for k in visits}
    all_physical={physical(x['row']) for x in data['train']['sft']}
    if set(visits)!=set(contexts) or observed!=all_physical:
        raise ValueError('450-update microbatch schedule loses training contexts/targets')
    groups=paired_groups(pairs)
    if steps<len(groups):raise ValueError('Update budget cannot cover both contexts of every preference group')
    budgets=lambda row:tuple(row['budget_frac_'+k] for k in ('bram_18k','dsp','ff','lut'))
    def summarize(rows):
        return {'kernels':sorted({x['row']['kernel_name'] for x in rows}),
                'physical_targets':len({physical(x['row']) for x in rows}),
                'budget_contexts':len({physical(x['row'])+(budgets(x['row']),) for x in rows})}
    sft,pair_rows,_=validation
    heldout=summarize(sft)
    all_val=summarize(data['val']['sft'])
    if set(heldout['kernels'])!=set(all_val['kernels']) or heldout['physical_targets']!=all_val['physical_targets']:
        raise ValueError('Validation selection drops kernels or device/clock targets')
    return {'schema':'film-gmt-exposure-coverage-v1','updates':steps,'sft_microbatches':steps*grad_accum,
        'train':{**summarize(data['train']['sft']),'sft_exposures_in_bank':len(data['train']['sft']),
                 'distinct_contexts_visited':len(visits),'minimum_context_visits':min(visits.values()),
                 'maximum_context_visits':max(visits.values()),'per_kernel_contexts':dict(collections.Counter(contexts[k][0]['row']['kernel_name'] for k in contexts)),
                 'preference_groups':len(groups),'preference_groups_by_kind':dict(collections.Counter(k[0] for k in groups)),
                 'both_contexts_per_preference_visit':True,'all_preference_groups_visited':True},
        'validation':{'full_bank':all_val,'selected_sft':heldout,'sft_examples':len(sft),
                      'pair_groups':len(pair_rows)//2,'pair_examples':len(pair_rows)},
        'selected_context_order_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest()}


def token_audit(data,parent,contract,registry,max_length):
    from LLM_branch.inference import eval_stage1_stage2_stage3 as inference
    from LLM_branch.train import train_SFT_xattn_new as prod
    tok=inference.load_checkpoint_tokenizer(str(parent),contract)
    code_cache={};target_cache={};worst=None;lengths={}
    for fold in ['train','val']:
        for kind in ['sft','pairs']:
            for index,entry in enumerate(data[fold][kind]):
                row=entry['row'];targets=[row['target']] if kind=='sft' else [entry['preferred_target'],entry['rejected_target']]
                header,code,suffix,_=prod.build_prompt_sections(row['input'],row['obj_mode'],row)
                if code not in code_cache:code_cache[code]=len(tok(code,add_special_tokens=False)['input_ids'])
                prompt=sum(len(tok(text,add_special_tokens=False)['input_ids']) for text in [header,suffix])+code_cache[code]
                for target in targets:
                    key=(row['kernel_name'],target)
                    if key not in target_cache:
                        core=prod.reorder_target_by_source_order(row['input'],target)
                        target_cache[key]=len(prod.build_deterministic_rhs_pack(row['input'],core,tok,
                            directive_domain_registry=registry,kernel_name=row['kernel_name']).input_ids)
                    total=prompt+len(prod.build_clock_pack(row,tok).input_ids)+target_cache[key]
                    if total>max_length:raise ValueError(f'Complete source would be truncated: {key[0]} {total}>{max_length}')
                    lengths[row['kernel_name']]=max(total,lengths.get(row['kernel_name'],0))
                    if fold=='train' and kind=='sft' and (worst is None or total>worst['tokens']):
                        worst={'fold':fold,'kind':kind,'index':index,'tokens':total,'kernel':row['kernel_name']}
    return {'status':'NO_TRUNCATION','max_length':max_length,'per_kernel_max_tokens':lengths,
            'longest_training_sft':worst,'max_tokens':max(lengths.values())}
