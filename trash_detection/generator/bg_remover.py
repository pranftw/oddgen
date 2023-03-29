import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.jit.load('bg_remover_models/isnet_scripted_reduction_gpu.pt').to(device)
model.load_state_dict(torch.load('bg_remover_models/isnet-general-use.pth'))