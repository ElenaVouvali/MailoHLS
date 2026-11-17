#!/usr/bin/env python3
"""Synthetic HLS mock for LOCAL TESTS ONLY; never use as DAC measurement."""
import pathlib
import re
import sys

if '-version' in sys.argv:
    print('MOCK Vitis HLS 2021.1 TEST ONLY')
    sys.exit(0)
p=pathlib.Path.cwd()
src=(p/'candidate.cpp').read_text()
tcl=(p/'run_hls.tcl').read_text()
top=re.search(r'set_top\s+(\w+)',tcl).group(1)
part=re.search(r'set_part\s+\{?([^}\s]+)',tcl).group(1)
clock=float(re.search(r'create_clock -period\s+(\S+)',tcl).group(1))
cyclic='type=cyclic factor=4' in src
latency=800 if cyclic else 1000
out=p/'hls_project'/'solution1'/'syn'/'report'
out.mkdir(parents=True)
(out/(top+'_csynth.xml')).write_text('''<profile><UserAssignments><TopModelName>%s</TopModelName><Part>%s</Part><TargetClockPeriod>%s</TargetClockPeriod></UserAssignments><PerformanceEstimates><SummaryOfTimingAnalysis><EstimatedClockPeriod>7.0</EstimatedClockPeriod></SummaryOfTimingAnalysis><SummaryOfOverallLatency><Worst-caseLatency>%d</Worst-caseLatency></SummaryOfOverallLatency><SummaryOfLoopLatency><L2><Latency>700</Latency><TripCount>16</TripCount><IterationInterval>4</IterationInterval></L2></SummaryOfLoopLatency></PerformanceEstimates><AreaEstimates><Resources><BRAM_18K>20</BRAM_18K><DSP>10</DSP><FF>300</FF><LUT>400</LUT></Resources></AreaEstimates></profile>'''%(top,part,clock,latency))
if not cyclic:
    line=next(i for i,s in enumerate(src.splitlines(),start=1) if 'acc += A[j]' in s)
    print("WARNING: [SCHED 69] Unable to schedule 'load' operation ('A_load', candidate.cpp:%d) on array 'A' due to limited memory ports."%line)
    print('INFO: [SCHED 61] Pipelining result: Target II: 1, Final II: 4, Depth: 6.')
print('Finished Command csynth_design')
print('Exiting vitis_hls')
