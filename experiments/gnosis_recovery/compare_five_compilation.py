#!/usr/bin/env python3
"""Frontend-only input comparison; never invokes vitis_hls or synthesis.

Run on a licensed Vitis 2021.1 installation with paired inputs/{original,canonical}.
Flags below come from a retained 2021.1 frontend invocation, NOT an original
GNOSIS per-measurement log. The distinction is recorded in the receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    root = Path(args.vitis_root)
    compiler = root / 'lnx64/tools/clang-3.9-csynth/bin/clang'
    disassembler = compiler.with_name('llvm-dis')
    env = dict(os.environ, LD_LIBRARY_PATH=str(root / 'lib/lnx64.o'))
    flags = ['-gcc-toolchain', str(root / 'tps/lnx64/gcc-6.2.0'), '-fhls',
             '-fno-exceptions', '-I'+str(root/'include'), '-fno-math-errno',
             '-fno-threadsafe-statics', '-fno-use-cxa-atexit', '-std=gnu++14',
             '-target', 'fpga64-xilinx-linux-gnu',
             '-I'+str(root/'common/technology/autopilot/39'),
             '-D__VITIS_HLS__', '-DAESL_SYN', '-D__SYNTHESIS__', '-D__HLS_SYN__',
             '-I'+str(root/'common/technology/autopilot'),
             '-I'+str(root/'common/technology/autopilot/ap_sysc'),
             '-include', 'etc/autopilot_ssdm_op.h', '-D__DSP48E2__',
             '-I/usr/include/x86_64-linux-gnu',
             '-hls-platform-db-name='+str(root/'common/technology/xilinx/common/platform.db'),
             '-hls-platform-name=zynquplus_medium']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='gnosis-five-frontend-comparison-v1',
                   historical_per_measurement_flags_available=False,
                   flags_basis='Retained coroni Vitis 2021.1 frontend log; corroboration only',
                   compiler=str(compiler), compiler_sha256=digest(compiler), flags=flags,
                   disassembler_sha256=digest(disassembler),
                   version=subprocess.run([str(compiler), '--version'], env=env,
                                          capture_output=True, text=True).stdout,
                   inputs={}, commands=[])
    headers = sorted((root/'include').rglob('*.h'))
    receipt['installed_header_sha256'] = {str(p.relative_to(root)):digest(p) for p in headers}
    def command(argv, stdout):
        result = subprocess.run(argv, env=env, capture_output=True, timeout=90)
        stdout.write_bytes(result.stdout)
        stderr = stdout.with_suffix(stdout.suffix+'.stderr')
        stderr.write_bytes(result.stderr)
        receipt['commands'].append(dict(argv=argv, returncode=result.returncode,
            stdout=str(stdout), stdout_sha256=digest(stdout),
            stderr=str(stderr), stderr_sha256=digest(stderr)))
    for kind in ['original', 'canonical']:
        for folder in sorted((Path(args.inputs)/kind).iterdir()):
            source, = folder.glob('*.cpp')
            key = kind+'/'+folder.name
            receipt['inputs'][key] = {p.name:digest(p) for p in sorted(folder.iterdir())}
            dest = output/kind/folder.name
            dest.mkdir(parents=True, exist_ok=True)
            for mode, extra in [('preprocessed', ['-E', '-P']),
                                ('macros', ['-E', '-dM']),
                                ('llvm', ['-S', '-emit-llvm', '-O0', '-o', '-'])]:
                command([str(compiler), *flags, *extra, str(source.resolve())], dest/(mode+'.txt'))
            # This vendor frontend emits bitcode even with -S; use its matching reader.
            command([str(disassembler), str(dest/'llvm.txt'), '-o', '-'], dest/'llvm.ll')
            if folder.name == 'rodinia-knn-5-coalescing':
                # A COUNTERFACTUAL branch probe, never claimed as historical flags.
                command([str(compiler), *flags, '-D__VIVADO_HLS__', '-E', '-P',
                         str(source.resolve())], dest/'counterfactual_vivado_preprocessed.txt')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default='inputs')
    parser.add_argument('--output', default='results')
    parser.add_argument('--vitis-root', default='/opt/Xilinx/Vitis_HLS/2021.1')
    run(parser.parse_args())
