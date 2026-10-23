import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for p in sorted((root/'results').glob('*/result.json')):
 r=json.loads(p.read_text()); inp=p.parent/'input.json'
 if inp.exists(): r['input_sha256']=json.loads(inp.read_text())['input_sha256']
 rows.append(r)
(root/'probe_results.json').write_text(json.dumps(rows,indent=2)+'\n')
compact=[]
for r in rows:
 compact.append({k:r.get(k) for k in ['candidate','kernel','stage','objective','target_context','status','latency','latency_source','resources','resources_source','synthesis_latency_cycles','synthesis_status','cosimulation_status','implementation_status','directives']})
(root/'SALT_final_probes.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in compact))
lines=['# Final SALT probes','', 'Target: ZCU104 (`xczu7ev-ffvc1156-2-e`), 10 ns, full budget. Resources are HLS synthesis estimates; latency is minimum Verilog/xsim co-simulation cycles for the supplied test input. No place-and-route implementation was run.','', '| Candidate | Status | Co-sim cycles | Synthesis min / max | BRAM_18K | DSP | FF | LUT |', '|---|---|---:|---|---:|---:|---:|---:|']
for r in rows:
 a=r.get('resources',{});s=r.get('synthesis_latency_cycles',{})
 lines.append('| %s | %s | %s | %s / %s | %s | %s | %s | %s |'%(r['candidate'],r['status'],r.get('latency'),s.get('min'),s.get('max'),a.get('bram'),a.get('dsp'),a.get('ff'),a.get('lut')))
lines+=['','These are manual follow-up probes, not new model predictions. Preserve the original SALT rows when reporting model-only performance. The GAN comparison retains the local arithmetic compatibility header used by its SALT reference builds.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
