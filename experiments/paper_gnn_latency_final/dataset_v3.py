"""Final matched loader: complete static source slots and four requested validation kernels."""
import os
from pathlib import Path
from experiments.paper_gnn.data import PaperData,sha,fingerprint
from experiments.paper_gnn_latency_final.evaluation_cohort import include_radix_ignoring_l6

class CompletePaperData(PaperData):
    def __init__(self,out):
        super().__init__(out)
        amendment=include_radix_ignoring_l6(self,Path(os.environ['SALT_CORRECTED_COHORT']))
        self.identity['validation_amendment']=amendment
        self.identity['loader_source_sha256']=sha(Path(__file__))
        self.coverage['accepted_unique_observed_development_configurations']=sum(len(v) for v in self.groups.values() if v[0]['split']=='val')
        self.identity['coverage']=self.coverage
