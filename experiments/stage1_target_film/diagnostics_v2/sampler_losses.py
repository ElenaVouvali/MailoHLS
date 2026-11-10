"""Exact near-optimal pair-sampler losses; never attribute overlapping stages twice."""
import collections,json
from pathlib import Path
from experiments.stage1_target_film.train import write
source=Path(__file__).with_name('qualify_clock.py').read_text().split('selected=[]')[0];source=source.replace('if dest.exists():raise FileExistsError(dest)','').replace('dest.mkdir(parents=True)','');ns={'__name__':'sampler_loss_reuse'};exec(compile(source,'qualified_clock_contrast_reuse','exec'),ns);contrasts=ns['contrasts'];result={'schema':'source-legal-clock-sampler-losses-v1','source':'same source-qualified clock audit, no locked test or GAN','by_split':{}}
for split in ('train','val'):
 rows=[r for r in contrasts if r['split']==split and r['status']=='switch'];result['by_split'][split]={}
 for name,rs in [('legal_clock_reversals_before_near_optimal_filter',rows),('lost_by_requiring_pair_winner_near_optimal_in_both_contexts',[r for r in rows if not r['near_optimal_both_contexts']]),('retained_after_near_optimal_filter',[r for r in rows if r['near_optimal_both_contexts']])]:
  result['by_split'][split][name]={'views':len(rs),'unique_candidate_pairs':len({(r['kernel'],r['a'],r['b']) for r in rs}),'independent_kernels':len({r['kernel'] for r in rs}),'independent_families':sorted({r['family'] for r in rs})}
result['other_stage_losses']='One-pair hashing, family balancing, source/action restoration and source legality overlap with the near-optimal filter; their separate causal loss counts are not identified by this counterfactual. Do not double-count.'
write(Path('experiments/stage1_target_film/diagnostics_v2/SAMPLER_LOSSES.json'),result);print(json.dumps(result,indent=2))
