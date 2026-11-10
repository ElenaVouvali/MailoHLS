"""Same prespecified optimizer/loss with qualified complete source-slot tensor inputs."""
from experiments.paper_gnn_latency_final import train
from experiments.paper_gnn_latency_final.memory_export import export
from experiments.paper_gnn_latency_final.dataset_v3 import CompletePaperData
train.export=export
train.PaperData=CompletePaperData
_original_backend=train.backend

def _idle_gpu_backend():
    # Recheck after CPU ingestion, immediately before CUDA model allocation.
    train.gpu_receipt('V100')
    return _original_backend()
train.backend=_idle_gpu_backend
if __name__=='__main__':raise SystemExit(train.main())
