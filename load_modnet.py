import torch
from modnet import MODNet  # make sure this matches your file name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MODNet with backbone_pretrained=False to avoid conflicts
modnet = MODNet(backbone_pretrained=False).to(device)
modnet.eval()

# Load official pretrained checkpoint
checkpoint_path = "pretrained/modnet_photographic_portrait_matting.ckpt"
state_dict = torch.load(checkpoint_path, map_location=device)
modnet.load_state_dict(state_dict, strict=False)  # ignore missing keys

print("MODNet loaded successfully!")


