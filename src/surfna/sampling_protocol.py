"""Frozen Figure 3 random-stream and strict-sampling rules."""
import ast,hashlib
def seed_for(master, complex_name, stage):
    key = f'surfna-figure3-k10-common-rng-v1|{master}|{complex_name}|{stage}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big')

def strict_sampling_ast(source):
    """Only remove swallowing handlers; retain the pinned diffusion arithmetic."""
    module = ast.parse(source)
    fun = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'sampling')
    fun.decorator_list = []
    changed = 0
    for node in ast.walk(fun):
        if not isinstance(node, ast.ExceptHandler) or node.type is not None:
            continue
        if len(node.body) != 1 or 'new_data_list.append(complex_graph)' != ast.unparse(node.body[0]):
            raise RuntimeError('Unexpected bare exception handler in pinned sampler')
        node.body = [ast.Raise()]
        changed += 1
    if changed != 1:
        raise RuntimeError(f'Expected exactly one swallowed conformer exception; got {changed}')
    return ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[]))

class NoiseTorch:
    """Private CPU Gaussian stream: unaffected by model/loader global RNG use."""
    def __init__(self, torch, seed):
        self.torch = torch
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.values = []

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def normal(self, *args, **kwargs):
        assert 'generator' not in kwargs
        value = self.torch.normal(*args, **kwargs, generator=self.generator)
        self.values.append(value.detach().cpu().numpy().copy())
        return value
