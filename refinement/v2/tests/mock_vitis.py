#!/usr/bin/env python3
"""Synthetic v2 control-flow fixture; not a hardware measurement."""
from pathlib import Path
import re

root = Path.cwd()
source = (root/'candidate.cpp').read_text()
tcl = (root/'run_hls.tcl').read_text()
top = re.search(r'set_top\s+(\w+)',tcl).group(1)
part = re.search(r'set_part\s+\{?([^}\s]+)',tcl).group(1)
clock = re.search(r'create_clock -period\s+(\S+)',tcl).group(1)
cyclic = 'type=cyclic factor=4' in source
best,worst = (800,1100) if cyclic else (1000,1200)
report = root/'hls_project/solution1/syn/report'/(top+'_csynth.xml')
report.parent.mkdir(parents=True)
report.write_text('''<profile><ReportVersion><Version>2021.1</Version></ReportVersion>
<UserAssignments><TopModelName>{top}</TopModelName><Part>{part}</Part><TargetClockPeriod>{clock}</TargetClockPeriod></UserAssignments>
<PerformanceEstimates><SummaryOfTimingAnalysis><EstimatedClockPeriod>7.0</EstimatedClockPeriod></SummaryOfTimingAnalysis>
<SummaryOfOverallLatency><Best-caseLatency>{best}</Best-caseLatency><Worst-caseLatency>{worst}</Worst-caseLatency></SummaryOfOverallLatency>
<SummaryOfLoopLatency><L2><Latency>700</Latency><IterationInterval>4</IterationInterval></L2></SummaryOfLoopLatency></PerformanceEstimates>
<AreaEstimates><Resources><BRAM_18K>20</BRAM_18K><DSP>10</DSP><FF>300</FF><LUT>400</LUT></Resources></AreaEstimates></profile>'''.format(top=top,part=part,clock=clock,best=best,worst=worst))
if not cyclic:
    line = next(i for i,text in enumerate(source.splitlines(),1) if 'acc += A[j]' in text)
    print("WARNING: [SCHED 69] Unable to schedule load (candidate.cpp:{}) on array 'A' due to limited memory ports (loop 'L2').".format(line))
    print("INFO: [SCHED 61] Pipelining result : Target II = 1, Final II = 4, Depth = 6, loop 'L2'")
print('Finished Command csynth_design')
print('Exiting vitis_hls')
