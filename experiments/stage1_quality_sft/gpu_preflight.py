"""Fail-closed check for the assigned, idle V100S before any training launch."""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess


def check(expected_uuid=None):
    expected_uuid = expected_uuid or os.environ.get('MAILOHLS_STAGE1_GPU_UUID')
    if not expected_uuid:
        raise RuntimeError('Set MAILOHLS_STAGE1_GPU_UUID to the documented assigned device UUID')
    query = ['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(query, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'NVIDIA environment unavailable: {exc}') from exc
    devices = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    found = [row for row in devices if row and row[0].strip() == expected_uuid]
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one visible assigned UUID {expected_uuid}; found {len(found)}')
    uuid, name, used, total, utilization = [x.strip() for x in found[0]]
    if 'V100' not in name.upper():
        raise RuntimeError(f'Assigned UUID is not a V100S-family device: {name}')
    try:
        memory_used, gpu_util = int(float(used)), int(float(utilization))
    except ValueError as exc:
        raise RuntimeError('Could not parse nvidia-smi utilization status') from exc
    if memory_used > 512 or gpu_util > 5:
        raise RuntimeError(f'Assigned V100S is occupied: memory={memory_used} MiB utilization={gpu_util}%')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
                          check=True, capture_output=True, text=True)
    owners = [line.strip() for line in apps.stdout.splitlines() if line.strip().startswith(expected_uuid)]
    if owners:
        raise RuntimeError(f'Assigned GPU has active compute processes: {owners}')
    return dict(uuid=uuid, name=name, memory_used_mib=memory_used, memory_total_mib=total,
                gpu_utilization_percent=gpu_util, compute_processes=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-uuid')
    args = parser.parse_args()
    print(check(args.expected_uuid))


if __name__ == '__main__':
    main()
