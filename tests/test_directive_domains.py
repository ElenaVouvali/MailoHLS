"""Exact uncapped domains and explicit opt-in workflow; no GPU/model needed."""
import copy
import json
from types import ModuleType

import pytest

from LLM_branch.common import directive_domains as legacy
from LLM_branch.common.directive_domains import tcad
from LLM_branch.common import directive_domain_workflow as workflow
from experiments.refinement import configurations as c


@pytest.mark.parametrize(
    'size,expected',
    [(30,[2,3,4,5,6,8,10,15,16,30]),(50,[2,4,5,8,10,16,25,32,50]),(32,[2,4,8,16,32]),(1,[])]
)
def test_exact_formula(size,expected):
    assert list(tcad.factor_values(size))==expected


@pytest.mark.parametrize('n',[3,7,9,30,50,65,1025,4096,10**6])
def test_factors_no_hidden_caps(n):
    values=tcad.factor_values(n)
    assert values==tuple(sorted(set(values)))
    assert all(2<=f<=n and (f & (f-1)==0 or n%f==0) for f in values)
    assert n in values
    assert all(f in values for f in range(2,min(n,5000)+1) if f & (f-1)==0 or n%f==0)


def test_large_power_and_composite_efficient():
    assert tcad.factor_values(2**48)[-1]==2**48
    assert 10**10 in tcad.factor_values(10**10)
    for n in [0,-1,None,True,2.5]:
        with pytest.raises(ValueError):tcad.factor_values(n)


@pytest.fixture
def metadata(tmp_path):
    (tmp_path/'kernel_info.txt').write_text('top\nL1,loop,30\nL9,array,a,1,30,2,50\n')
    actions=legacy.load_source_actions('toy',tmp_path)
    text='\n'.join(c.ordered_fields(actions))
    domains=tcad.source_site_domains('toy',text,tmp_path)
    baseline={k:'none' if 'ARRAY_T' in k else '0' for k in domains}
    return tmp_path,actions,text,domains,baseline


def test_dimension_specific_validation_and_disabled_semantics(metadata):
    _,actions,_,_,baseline=metadata
    values={**baseline,c.field_key('PIPE','L1'):'1',c.field_key('UNROLL','L1'):'15'}
    tcad.validate_assignments(values,actions)
    for typ,factor,dim,valid in [
        ('none','0','0',True),
        ('complete','0','2',True),
        ('block','6','1',True),
        ('cyclic','25','2',True),
        ('cyclic','6','2',False),
        ('block','32','1',False),
        ('none','0','1',False),
        ('complete','2','1',False),
        ('block','0','1',False)
    ]:
        row={
            **values,
            c.field_key('ARRAY_T','L9'):typ,
            c.field_key('ARRAY_F','L9'):factor,
            c.field_key('ARRAY_D','L9'):dim
        }
        if valid:assert not tcad.validate_assignments(row,actions)
        else:
            with pytest.raises(ValueError):tcad.validate_assignments(row,actions)


def test_sequential_filter_factor_before_dimension(metadata):
    _,_,_,domains,_=metadata
    typ,factor,dim=(c.field_key(k,'L9') for k in ['ARRAY_T','ARRAY_F','ARRAY_D'])
    assert '25' in tcad.filter_semantic_candidates(factor,domains[factor],{typ:'cyclic'},domains)
    assert tcad.filter_semantic_candidates(dim,domains[dim],{typ:'cyclic',factor:'25'},domains)==['2']
    assert tcad.filter_semantic_candidates(dim,domains[dim],{typ:'block',factor:'6'},domains)==['1']
    assert tcad.filter_semantic_candidates(factor,domains[factor],{typ:'none'},domains)==['0']
    with pytest.raises(ValueError):tcad.filter_semantic_candidates(dim,['1'],{typ:'block',factor:'25'},domains)
    with pytest.raises(ValueError):tcad.filter_semantic_candidates(dim,domains[dim],{},dict(domains))


def test_historical_exception_cannot_license_new_edits(metadata):
    _,actions,_,_,baseline=metadata
    old={**baseline,c.field_key('UNROLL','L1'):'7'}
    with pytest.raises(ValueError):tcad.validate_assignments(old,actions)
    assert 'L1' in tcad.validate_assignments(old,actions,old,True)
    with pytest.raises(ValueError):tcad.validate_assignments({**old,c.field_key('PIPE','L1'):'1'},actions,old,True)
    with pytest.raises(ValueError):tcad.validate_assignments({**old,c.field_key('UNROLL','L1'):'9'},actions,old,True)


def test_unresolved_metadata_is_not_fabricated(metadata):
    path,*_=metadata
    action=legacy.SourceAction(kind='loop',trip_count=None)
    assert tcad.action_tuples(action)==(('0','0'),('1','0'))
    (path/'kernel_info.txt').write_text('top\nL1,loop,unknown\n')
    with pytest.raises(ValueError):tcad.load_source_actions('toy',path)


def test_legacy_unchanged_and_registry_source_metadata_retained(metadata):
    path,actions,text,_,_=metadata
    old=legacy.source_site_domains('toy',text,path)
    saved=copy.deepcopy(old)
    registry=tcad.build_source_domain_registry([{'kernel_name':'toy','input':text}],path)
    assert isinstance(registry['toy'],tcad.SourceDomains)
    assert registry['toy'].actions==actions
    assert '6' not in old[c.field_key('ARRAY_F','L9')] and '6' in registry['toy'][c.field_key('ARRAY_F','L9')]
    assert old==saved and legacy.SOURCE_DOMAIN_POLICY=='source_action_metadata_proposal_domains_v2'
    assert legacy.MAX_UNROLL_FACTOR==64 and legacy.MAX_PARTITION_FACTOR==1024


def fake_module():
    module=ModuleType('fixture_workflow')
    exec('''
def constrained(source_text, kernel_name, directive_domain_registry=None):
    domains=directive_domain_registry[kernel_name]
    return {'domains':domains,'allowed':mailohls_contract.filter_semantic_candidates('AUTO{_ARRAY_D_L9}', ['1','2'], {'AUTO{_ARRAY_T_L9}':'cyclic','AUTO{_ARRAY_F_L9}':'25'}, domains)}
def load_directive_domain_registry(path):
    import json
    return json.load(open(path))['domains']
''',module.__dict__)
    module.directive_domains=legacy
    return module


def test_opt_in_decoder_scoping_and_checkpoint_disclosure(metadata):
    path,_,text,_,_=metadata
    module=fake_module()
    with pytest.raises(ValueError,match='domain-shift'):
        workflow.install_policy(module,path,workflow='infer',trained_policy=legacy.SOURCE_DOMAIN_POLICY)
    workflow.install_policy(
        module,
        path,
        workflow='infer',
        trained_policy=legacy.SOURCE_DOMAIN_POLICY,
        allow_domain_shift=True
    )
    result=module.constrained(text,'toy')
    assert result['allowed']==['2'] and result['domain_shift_evaluation']
    assert result['checkpoint_domain_policy']==legacy.SOURCE_DOMAIN_POLICY
    assert result['evaluation_domain_policy']==tcad.SOURCE_DOMAIN_POLICY
    assert module.directive_domains is legacy and workflow.ACTIVE.get() is None


def test_explicit_tcad_train_registry_and_zero_caps(metadata,tmp_path):
    path,_,text,domains,_=metadata
    module=fake_module()
    workflow.install_policy(module,path,workflow='train')
    registry=tmp_path/'registry.json'
    registry.write_text(json.dumps({'generation_policy':tcad.SOURCE_DOMAIN_POLICY,'domains':{'toy':domains}}))
    assert isinstance(module.load_directive_domain_registry(registry)['toy'],tcad.SourceDomains)
    assert module.directive_domains.MAX_UNROLL_FACTOR is None
    assert module.constrained(text,'toy')['allowed']==['2']
    registry.write_text(json.dumps({'generation_policy':legacy.SOURCE_DOMAIN_POLICY,'domains':{'toy':domains}}))
    with pytest.raises(ValueError,match='relabeled'):module.load_directive_domain_registry(registry)
