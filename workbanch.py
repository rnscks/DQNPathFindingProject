import torch

# 3차원 텐서 생성
tensor = torch.zeros((10, 10, 10))
print(tensor)
# 배치 차원과 채널 차원 추가
tensor = tensor.view(1, 1, *tensor.shape)  # (1, 1, 깊이, 높이, 너비)

print(tensor)