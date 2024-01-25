import torch as tr
# from torchsummary import summary

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

def init_nerf_model(D=8, W=256, input_ch=3, output_ch=6, 
                    skips=[4], use_viewdirs=False):
    # print('MODEL', input_ch, type(input_ch), use_viewdirs)
    "MODEL 63 <class 'int'> False"
    
    input_ch = int(input_ch)
    
    input_size = input_ch+3 
    
    inp = tr.nn.Parameter(tr.empty(input_size))
    inp_pts, inp_views = tr.split(inp, [input_ch, 3], -1)
    
    inputs_pts = inp_pts.view(-1, input_ch)
    
    # print("input : {}".format(inputs_pts.shape))
    "input : torch.Size([1, 63])"
    
    class USNeRF_Model(tr.nn.Module):
        def __init__(self, D, W, skips, start_in):
            super(USNeRF_Model, self).__init__()
            self.D = D
            self.W = W
            self.skips = skips
            self.start_in = start_in
            
            self.first_layer = tr.nn.Linear(self.start_in, self.W)
            self.dense = tr.nn.Linear(self.W, self.W)
            self.relu = tr.nn.ReLU()
            
            layers = [self.first_layer, self.relu]
            
            for i in range(self.D-2):
                layers += [self.dense, self.relu]
            
            layers += [tr.nn.Linear(self.W, 128)]
            
            self.model = tr.nn.Sequential(*layers)
        def forward(self, x):
            
            return self.model(x)
    
    USNM = USNeRF_Model(D, W, skips, input_ch)
    USNM.to(device)
    # print(next(USNM.parameters()).device) # 應該要是 cuda:0
    
    # print(USNM)
    # summary(USNM, input_ch)
    
    return USNM
               
def img2mse(x, y): return tr.mean(tr.square(x-y))       