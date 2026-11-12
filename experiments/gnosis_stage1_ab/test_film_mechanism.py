"""CPU unit tests for the FiLM part; no MailoHLS or GPU required."""
import ast,contextlib,math,re,unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

src=Path(__file__).with_name('train_gnosis_ab.py').read_text()
tree=ast.parse(src)
body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('ResidualFiLM','InstalledFiLM')]
g={'torch':torch,'nn':nn,'F':F,'contextlib':contextlib,'__name__':'__test__'}
exec(compile(ast.Module(body=body,type_ignores=[]),'extracted_classes','exec'),g)
ResidualFiLM,InstalledFiLM=g['ResidualFiLM'],g['InstalledFiLM']

class Core(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList(nn.Linear(16,16) for _ in range(8))
class Wrapper(nn.Module):
    def __init__(self):super().__init__();self.core=Core()
    def get_base_model(self):return self.core
    def forward(self,x):
        for layer in self.core.model.layers:
            x=checkpoint(layer,x,use_reentrant=False)
        return x
class TestFiLM(unittest.TestCase):
    def test_shape_mask_context_difference_and_grads(self):
        torch.manual_seed(8);m=Wrapper();film=ResidualFiLM(16,8,bottleneck=12)
        hook=InstalledFiLM(m,film)
        x=torch.randn(1,5,16,requires_grad=True)
        kinds=torch.tensor([[0,1,2,3,0]])
        a=torch.rand(1,1,15);b=a.clone();b[:,:,4]+=0.8
        with hook.bind(a,kinds,enabled=True):
            y=m(x);y.square().mean().backward()
        self.assertIsNone(film.state)
        grads=[p.grad.abs().sum().item() for p in film.parameters() if p.grad is not None]
        self.assertGreater(max(grads),0)
        self.assertTrue(torch.isfinite(y).all())
        with torch.no_grad():
            with hook.bind(a,kinds,enabled=True):u=m(x.detach()).clone()
            with hook.bind(b,kinds,enabled=True):v=m(x.detach()).clone()
        self.assertGreater(float((u-v).norm()),1e-9)
    def test_off_and_inactive_are_identical(self):
        m=Wrapper();film=ResidualFiLM(16,8,bottleneck=12);hook=InstalledFiLM(m,film)
        x=torch.randn(1,4,16);kind=torch.zeros(1,4,dtype=torch.long);ctx=torch.zeros(1,1,15)
        with torch.no_grad():
            with hook.bind(ctx,kind,True):a=m(x)
            with hook.bind(ctx,kind,False):b=m(x)
        self.assertTrue(torch.equal(a,b))

if __name__=='__main__':unittest.main()
