import torch
import torch.nn as nn
import time


conv_2d = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1).cuda()
feat_2d = torch.randn(8, 256, 90, 300).cuda()
time_2d = 0
for _ in range(50):
    torch.cuda.synchronize()
    start = time.time()
    out = conv_2d(feat_2d)
    torch.cuda.synchronize()
    end = time.time()
    time_2d += end - start
time_2d /= 50
print("time_2d", time_2d) # 3ms
# conv_3d = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1).cuda()
# feat_3d = torch.randn(8, 32, 90, 300, 40).cuda()
# time_3d = 0
# for _ in range(10):
#     torch.cuda.synchronize()
#     start = time.time()
#     out = conv_3d(feat_3d)
#     torch.cuda.synchronize()
#     end = time.time()
#     time_3d += end - start
# time_3d /= 10

# print("time_3d", time_3d) # 63ms