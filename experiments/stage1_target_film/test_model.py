import torch
from .model import ActionFiLM,kind_id

def test_identity_gradients_cache_and_roundtrip():
 torch.manual_seed(123);m=ActionFiLM(16);h=torch.randn(1,6,16);z=torch.randn(1,1,15);k=torch.tensor([[0,0,1,2,3,5]])
 assert torch.equal(m(h,z,k),h)
 m(h,z,k).square().sum().backward();assert m.gate.grad.abs()>0
 assert m.net[-1].weight.grad.abs().sum()==0
 with torch.no_grad():m.gate.fill_(.01)
 m.zero_grad();out=m(h,z,k);out.square().sum().backward();assert m.net[-1].weight.grad.norm()>0
 assert torch.equal(out[:,:2],h[:,:2])
 assert torch.allclose(out[:,-1:],m(h[:,-1:],z,k[:,-1:]),atol=1e-7)
 clone=ActionFiLM(16);clone.load_state_dict(m.state_dict());assert torch.equal(out,clone(h,z,k))
 assert not torch.equal(out,m(h,z+1,k))
 assert [kind_id('AUTO{_'+x+'_L63}') for x in ('PIPE','UNROLL','ARRAY_T','ARRAY_F','ARRAY_D')]==[1,2,3,4,5]
