model.load_state_dict(torch.load("model_100.pth", map_location=device, weights_only=True))
model.eval() # dropout을 사용하지 않음