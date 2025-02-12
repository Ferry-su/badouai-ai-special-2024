import torch

# 自定义一个线性层（全连接层）
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        # 初始化方法，定义了该层的输入特征数、输出特征数以及是否使用偏置项
        super(Linear, self).__init__()

        # 使用 torch.nn.Parameter 创建权重参数，形状为 (out_features, in_features)
        # out_features 是该层的输出特征数，in_features 是输入特征数
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

        # 如果需要偏置项，则初始化偏置，形状为 (out_features)
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None  # 如果没有偏置项，则将 bias 设为 None

    def forward(self, x):
        # 前向传播方法，定义了数据如何通过该层进行计算
        # x 是输入数据，形状为 (batch_size, in_features)
        # 需要将输入数据 x 和权重矩阵 self.weight 进行矩阵乘法
        # 注意：要对权重矩阵进行转置，因为矩阵乘法要求 (batch_size, in_features) 和 (in_features, out_features) 的形状匹配
        x = x.mm(self.weight.t())  # weight.t() 是转置操作

        # 如果使用了偏置项，将偏置添加到每个输出特征上
        if self.bias is not None:  # 修改这里
            # expand_as 用来将偏置扩展为与输入的 x 相同的形状，以便能够进行加法
            x = x + self.bias.expand_as(x)

        return x  # 返回计算结果


if __name__ == '__main__':
    # 创建一个 Linear 层，输入特征为 3，输出特征为 2
    net = Linear(3, 2)

    # 创建一个输入数据 x，形状为 (1, 3)，表示一个样本，每个样本有 3 个特征
    input_data = torch.randn(1, 3)  # 使用 torch.randn 随机生成输入数据，1个样本，每个样本3个特征

    # 前向传播，计算输出
    output = net(input_data)  # 调用 net 的 forward 方法，传入 input_data 计算输出结果

    # 打印输出结果
    print('Output:', output)  # 输出结果形状为 (1, 2)，即每个样本有 2 个输出特征