"""Bound temporary attention memory without dropping points, keys, or views."""
import torch,inspect,textwrap

def attention(self,q,k,v,dtype,point=None,order=None):
    assert not self.training and not self.enable_rpe
    outputs=[]
    for b in range(q.shape[0]):
        rows=[]
        for start in range(0,q.shape[-2],256):
            scores=(q[b:b+1,:,start:start+256]*self.scale)@k[b:b+1].transpose(-2,-1)
            if self.upcast_softmax:scores=scores.float()
            probs=self.attn_drop(self.softmax(scores)).to(dtype)
            rows.append(probs@v[b:b+1])
        outputs.append(torch.cat(rows,dim=-2))
    return torch.cat(outputs,dim=0)

def install():
    from core.models.encoders.sonata import model as module
    cls=module.SerializedAttention
    source=textwrap.dedent(inspect.getsource(cls.forward))
    old='''        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        if self.enable_rpe:
            attn = attn + self.rpe(self.get_rel_pos(point, order))
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)'''
    assert source.count(old)==1
    source=source.replace(old,'        feat = _bounded_attention(self,q,k,v,qkv.dtype).transpose(1,2).reshape(-1,C)')
    namespace=dict(module.__dict__);namespace['_bounded_attention']=attention
    exec(compile(source,'<isolated_chunk_attention>','exec'),namespace)
    cls.forward=namespace['forward']

def validate():
    from types import SimpleNamespace
    torch.manual_seed(2)
    m=SimpleNamespace(training=False,enable_rpe=False,scale=32**-.5,upcast_softmax=True,softmax=torch.nn.Softmax(-1),attn_drop=torch.nn.Identity())
    reports=[]
    for dtype in (torch.float32,torch.bfloat16):
        q,k,v=[torch.randn(2,4,513,32,device='cuda',dtype=dtype) for _ in range(3)]
        with torch.autocast('cuda',enabled=dtype==torch.bfloat16,dtype=torch.bfloat16):
            expected=(m.softmax(((q*m.scale)@k.transpose(-2,-1)).float()).to(dtype)@v)
            actual=attention(m,q,k,v,dtype)
        diff=(actual.float()-expected.float()).abs()
        torch.testing.assert_close(actual,expected,rtol=1e-5 if dtype==torch.float32 else .01,atol=1e-6 if dtype==torch.float32 else .002)
        reports.append({'dtype':str(dtype),'max_abs':diff.max().item(),'mean_abs':diff.mean().item()})
    return reports
