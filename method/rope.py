# import torch

# # x = torch.tensor([1, 2, 3, 4, 5])
# # y = torch.tensor([10, 20, 30, 40, 50])

# # condition = x > 3

# # result = torch.where(condition, x, y)

# # print(result)

# # import torch

# # t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
# # t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])

# # # 沿着 dim=0 拼接 → 增加 batch 维度
# # result = torch.cat((t1, t2), dim=0)
# # print("沿 dim=0 拼接结果：")
# # print(result)
# # print()

# # # 沿着 dim=-1 拼接 → 增加最后一个维度（列方向）
# # result2 = torch.cat((t1, t2), dim=-1)
# # print("沿 dim=-1 拼接结果：")
# # print(result2)

# t1 = torch.Tensor([1, 2, 3])
# t2 = t1.unsqueeze(0)
# print(t1.shape)
# print(t2)
# print(t2.shape)