# Rosetta 3D Rendering
Uses Rosetta's OpenCL/HLS `rendering.cpp` top (`rendering`) and pinned `typedefs.h`. Existing PIPELINE/DATAFLOW/INLINE pragmas are removed so the external test does not inherit the benchmark's hand-tuned optimization policy; AXI/control INTERFACE pragmas are preserved. Local arrays and statically bounded loops become MailoHLS action candidates.
