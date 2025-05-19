from ptflops import get_model_complexity_info
import torchvision.models as models
import torch.nn as nn

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)

input_res = (3, 224, 224)

print("=== Layer-wise FLOPs and params ===")
for split_idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
    sub_model = nn.Sequential(*list(model.features)[:split_idx])
    macs, params = get_model_complexity_info(sub_model, input_res, as_strings=False, print_per_layer_stat=False)
    
    M = macs
    A = macs
    Gamma = params * 2
    
    print(f"Layer {split_idx}: M={M:.2e}, A={A:.2e}, Γ={Gamma:.2e}, Params={params:.2e}")
