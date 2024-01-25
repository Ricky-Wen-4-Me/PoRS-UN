import torch as tr

class Embedder:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        
        d = self.kwargs['input_dims']
        out_dim = 0
        
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        B = self.kwargs['B']
        
        if self.kwargs['log_sampling']:
            # **: 指數計算(運算的場合)
            freq_bands = 2. ** tr.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tr.linspace(2.**0., 2.**max_freq, N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if B is not None:
                    # @: Matrix Multiplication
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(x @ tr.transpose(B) * freq))
                    
                    out_dim += d
                    out_dim += B.shape[1]
                else:
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq, B=B: p_fn(x * freq))
                    
                    out_dim += d
            
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def embed(self, inputs):
        return tr.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, b=0):
    if i == -1:
        return tr.clone, 3
    if b != 0:
        #TODO: check seed
        B = tr.randn((b, 3), seed=1)
    else:
        B = None

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tr.sin, tr.cos],
        'B': B # for Gaussian
    }
    embedder_obj = Embedder(**embed_kwargs)
    
    def embed(x, eo=embedder_obj): return eo.embed(x)
    
    return embed, embedder_obj.out_dim