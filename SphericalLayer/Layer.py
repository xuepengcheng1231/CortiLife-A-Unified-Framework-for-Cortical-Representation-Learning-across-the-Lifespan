import math
import torch.nn as nn
import torch
from torch import Tensor
from util import utils


class onering_conv_layer(nn.Module):
    def __init__(self, in_features, out_features, neigh_orders):
        super(onering_conv_layer,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7*in_features,out_features)

    def forward(self,x):
        mat = x[self.neigh_orders]
        mat = mat.view(len(x),7*self.in_features)
        out_features = self.weight(mat)
        return out_features

class onering_conv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,neigh_orders):
        super(onering_conv_layer_batch,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7 * in_features, out_features)
    ## x.shape = N * features * vertices
    def forward(self,x):
        mat = x[:,:, self.neigh_orders]
        mat = mat.view(x.shape[0], self.in_features, x.shape[2],7).permute(0,2,3,1)
        mat = mat.contiguous().view(x.shape[0],x.shape[2],7*self.in_features)
        out_features = self.weight(mat).permute(0,2,1)
        return out_features

class OneRingConvFastLowMem(nn.Module):
    """
    最省显存且尽量快的实现：
    - 按 V 维分块处理，避免构造 (N,C,V,7) 的大张量
    - 不把 7 个邻居拼成 7C；改为对每个邻居单独做一次小型 matmul，然后累加到 out
    - 只在最外层预分配输出，避免中间反复申请内存
    - neigh_orders 作为 buffer，避免每次 forward 复制
    训练时建议外面套 AMP（bfloat16 优先，其次 float16）
    """
    def __init__(self, in_features: int, out_features: int, neigh_orders: Tensor,
                 chunk_size: int = 8192, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = int(chunk_size)

        # neigh_orders: (V, 7) 的 long 索引，注册成 buffer（随模型走 device）
        self.register_buffer("neigh_orders", torch.tensor(neigh_orders).long(), persistent=False)
        self.neigh_orders = self.neigh_orders.reshape([-1,7])
        # 为 7 个邻居各自准备一组权重： (7, out_features, in_features)
        # 这样可避免把特征拼成 7C，直接对每个邻居做一次小矩阵乘，再累加。
        self.weight = nn.Parameter(torch.empty(7, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # 参数初始化：跟 nn.Linear 一致的 kaiming_uniform
        for k in range(7):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        if bias:
            # 模仿 nn.Linear 的 fan_in 方式给 bias 合理范围
            fan_in = in_features * 7
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def _check_shapes(self):
        assert self.neigh_orders.dim() == 2 and self.neigh_orders.size(1) == 7, \
            "neigh_orders 应为 (V, 7)"
        assert self.weight.shape == (7, self.out_features, self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, C, V)
        return: (N, out_features, V)
        """
        # 可训练态下也要做一次（不加 no_grad）
        self._check_shapes()

        N, C, V = x.shape
        assert C == self.in_features, "输入通道不匹配 in_features"

        # 预分配输出（N, out, V）
        out = x.new_zeros((N, self.out_features, V))

        # 分块遍历 V，降低峰值显存
        for s in range(0, V, self.chunk_size):
            e = min(s + self.chunk_size, V)
            vchunk = e - s

            # 当前块的邻接索引：(vchunk, 7)
            idx_chunk = self.neigh_orders[s:e]  # long on same device

            # 对 7 个邻居逐个 gather & matmul，再累加到 out_chunk
            # out_chunk: (N, out, vchunk)
            out_chunk = out[:, :, s:e]  # 直接拿出视图，避免新分配

            # 循环 7 次小型 matmul（更省显存，速度也很可观）
            # x_k: (N, C, vchunk)     W_k: (out, C)
            # y_k = W_k @ x_k  -> (N, out, vchunk)
            for k in range(7):
                idx_k = idx_chunk[:, k]                      # (vchunk,)
                x_k = x.index_select(dim=2, index=idx_k)     # (N, C, vchunk)
                # einsum 比 reshape/matmul 更直观，并且避免额外拷贝：
                # (out, C) x (N, C, vchunk) -> (N, out, vchunk)
                y_k = torch.einsum('oc,ncv->nov', self.weight[k], x_k)
                out_chunk.add_(y_k)  # 原地累加，避免生成新张量

            # 加偏置（广播到 vchunk），放在 7 个邻居累加之后，少一次重复加法
            if self.bias is not None:
                out_chunk.add_(self.bias.view(1, -1, 1))

        return out

class pool_layer(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders = neigh_orders
        self.pool_type = pool_type
    # x.shape = N * output_features
    def forward(self,x):
        number_nodes = int((x.size()[0]+6)/4)
        features_num = x.size()[1]
        x = x[self.neigh_orders[0:number_nodes*7]].view(number_nodes,7,features_num)
        if self.pool_type == "mean":
            x=torch.mean(x,dim=1)
        if self.pool_type == "max":
            x = torch.max(x,dim=1)
            return x[0],x[1]
        return x
class pool_layer_batch(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders=neigh_orders
        self.pool_type=pool_type
    # x.shape = B * output_features * N
    def forward(self,x):
        number_nodes = int((x.size()[2]+6)/4)
        features_number = x.size()[1]
        x = x[:,:,self.neigh_orders[0:number_nodes*7]]
        x = x.view(x.size()[0],features_number,number_nodes,7)
        if self.pool_type == "mean":
            x = torch.mean(x, dim=3)
        if self.pool_type == "max":
            x = torch.max(x, dim=3)
            return x[0]
        return x

class pool_layer_batch_3D(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders=neigh_orders
        self.pool_type=pool_type
    # x.shape = B * output_features * N
    def forward(self,x):
        number_nodes = int((x.size()[3]+6)/4)
        length = x.size()[2]
        features_number = x.size()[1]
        x = x[:,:,:,self.neigh_orders[0:number_nodes*7]]
        x = x.view(x.size()[0],features_number,length,number_nodes,7)
        if self.pool_type == "mean":
            x = torch.mean(x, dim=4)
        if self.pool_type == "max":
            x = torch.max(x, dim=4)
            return x[0]
        return x

class upconv_layer(nn.Module):
    def __init__(self, in_features, out_features, upconv_center_indices, upconv_edge_indices):
        super(upconv_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upcon_center_indices = upconv_center_indices
        self.upcon_edge_indices = upconv_edge_indices
        self.weight = nn.Linear(in_features,7*out_features)
    # N*in_features
    def forward(self,x):
        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x)
        x = x.view(x.shape[0]*7, self.out_features)
        x1 = x[self.upcon_center_indices]
        assert (x1.size() == torch.Size([raw_nodes, self.out_features]))
        x2 = x[self.upcon_edge_indices].view(-1,self.out_features,2)
        x = torch.cat((x1,torch.mean(x2,dim=2)),0)
        assert(x.size() == torch.Size([new_nodes, self.out_features]))
        return x

class upconv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,upconv_center_indices,upconv_edge_indices):
        super(upconv_layer_batch,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upconv_center_indices = upconv_center_indices
        self.upconv_edge_indices = upconv_edge_indices
        self.weight = nn.Conv1d(in_features, 7 * out_features, kernel_size=1)
    # input N * vertices * features
    def forward(self,x):
        raw_nodes = x.size()[2]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x) # N * (7*out_features) * vertices
        x = x.permute(0,2,1)
        x = x.contiguous().view(x.shape[0],raw_nodes*7,self.out_features).permute(0,2,1)

        x1 = x[:, :, self.upconv_center_indices]
        assert (x1.size() == torch.Size([x.shape[0], self.out_features, raw_nodes]))
        x2 = x[:, :, self.upconv_edge_indices].view(x.shape[0], self.out_features, -1, 2)
        x = torch.cat((x1, torch.mean(x2, 3)), 2)
        assert (x.size() == torch.Size([x.shape[0], self.out_features, new_nodes]))
        # x = self.norm(x)
        return x

class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()

        self.conv1 = onering_conv_layer_batch(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = onering_conv_layer_batch(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block


    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.first:
            res = torch.cat((res, res), 1)
        x = x + res
        x = self.relu(x)

        return x

class OneRingConv3D(nn.Module):
    """
    Chunked one-ring spatiotemporal convolution.

    Input:
      - [B, T, V], treated as single-channel input
      - [B, C, T, V]

    Output:
      - [B, Cout, L, V]
    """

    def __init__(
        self,
        cin: int,
        cout: int,
        kt: int,
        neigh_orders,
        stride_t: int = 1,
        padding_t="same",
        bias: bool = True,
        vertex_chunk_size: int = 32,
    ):
        super().__init__()
        self.cin = int(cin)
        self.cout = int(cout)
        self.kt = int(kt)
        self.stride_t = int(stride_t)
        self.vertex_chunk_size = int(vertex_chunk_size)

        if self.vertex_chunk_size <= 0:
            raise ValueError("vertex_chunk_size must be > 0")

        if isinstance(padding_t, str):
            if padding_t.lower() != "same":
                raise ValueError(f"Unsupported padding_t string: {padding_t}")
            if self.kt % 2 != 1:
                raise ValueError("padding_t='same' requires odd kt.")
            self.pad_t = (self.kt - 1) // 2
        else:
            self.pad_t = int(padding_t)
            if self.pad_t < 0:
                raise ValueError("padding_t must be >= 0")

        neigh_orders = torch.as_tensor(neigh_orders, dtype=torch.long).view(-1, 7)
        self.register_buffer("neigh_orders", neigh_orders)

        self.conv = nn.Conv1d(
            in_channels=7 * self.cin,
            out_channels=self.cout,
            kernel_size=self.kt,
            stride=self.stride_t,
            padding=self.pad_t,
            bias=bias,
        )

    @staticmethod
    def _normalize_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
            return x
        raise ValueError(f"Expected input rank 3 or 4, got shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        bsz, cin, time_steps, num_vertices = x.shape

        if cin != self.cin:
            raise RuntimeError(f"Expected Cin={self.cin}, got {cin}")
        if num_vertices != self.neigh_orders.shape[0]:
            raise RuntimeError(
                f"Expected V={self.neigh_orders.shape[0]} from neigh_orders, got {num_vertices}"
            )

        outputs = []
        for start in range(0, num_vertices, self.vertex_chunk_size):
            end = min(start + self.vertex_chunk_size, num_vertices)
            chunk_neigh = self.neigh_orders[start:end]

            # [B, C, T, Vc, 7] -> [B*Vc, 7*C, T]
            chunk_x = x[..., chunk_neigh]
            chunk_x = chunk_x.permute(0, 3, 4, 1, 2).contiguous()
            chunk_x = chunk_x.view(bsz * (end - start), 7 * cin, time_steps)

            chunk_y = self.conv(chunk_x)
            out_time = chunk_y.shape[-1]

            # [B*Vc, Cout, L] -> [B, Cout, L, Vc]
            chunk_y = chunk_y.view(bsz, end - start, self.cout, out_time)
            chunk_y = chunk_y.permute(0, 2, 3, 1).contiguous()
            outputs.append(chunk_y)

        return torch.cat(outputs, dim=-1)

class res_block_3D(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders,kt,stride_t,padding_t,vertex_chunk_size, first_in_block=False, residual=True):
        super(res_block_3D, self).__init__()
        self.conv1 = OneRingConv3D(c_in, c_out, kt=kt,neigh_orders=neigh_orders,stride_t=stride_t,padding_t=padding_t,vertex_chunk_size=vertex_chunk_size)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = OneRingConv3D(c_out, c_out, kt=5,neigh_orders=neigh_orders,stride_t=1,padding_t=2,vertex_chunk_size=32)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.first = first_in_block
        self.residual = residual


    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.first and self.residual:
            res = res.view(res.shape[0], res.shape[1], res.shape[2]//2, 2, res.shape[3]).permute(0, 1, 3, 2, 4).contiguous().view(res.shape[0], res.shape[1]*2,-1, res.shape[3])
        if self.residual:
            x = x + res
        else:
            return self.relu(x)
        x = self.relu(x)

        return x

if __name__ == "__main__":
    _, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = utils.Get_neighs_order()
    model = OneRingConv3D(1,4,kt=4,neigh_orders=neigh_orders_40962,stride_t=2,padding_t=1,vertex_chunk_size=32)
    pool = pool_layer_batch_3D(neigh_orders_40962, 'mean')
    model1 = res_block_3D(4, 8, kt=4, neigh_orders=neigh_orders_10242, stride_t=2, padding_t=1, vertex_chunk_size=32,
                         first_in_block=True)
    # model2 = res_block_3D(4, 8, kt=4, neigh_orders=neigh_orders_10242, stride_t=2, padding_t=1, vertex_chunk_size=32,
    #                      first_in_block=True)
    # model3 = res_block_3D(1,4,kt=4,neigh_orders=neigh_orders_40962,stride_t=2,padding_t=1,vertex_chunk_size=32,first_in_block=True)
    model4= nn.Sequential(
        OneRingConv3D(1, 4, kt=4, neigh_orders=neigh_orders_40962, stride_t=2, padding_t=1, vertex_chunk_size=32),
        nn.BatchNorm2d(4),
        nn.LeakyReLU(0.2),
        res_block_3D(4, 4, kt=5, neigh_orders=neigh_orders_40962, stride_t=1, padding_t=2, vertex_chunk_size=32),
        pool_layer_batch_3D(neigh_orders_40962, 'mean'),
        res_block_3D(4, 8, kt=4, neigh_orders=neigh_orders_10242, stride_t=2, padding_t=1, vertex_chunk_size=32,
                     first_in_block=True),
        res_block_3D(8, 8, kt=5, neigh_orders=neigh_orders_10242, stride_t=1, padding_t=2, vertex_chunk_size=32),
        pool_layer_batch_3D(neigh_orders_10242, 'mean'),
        res_block_3D(8, 16, kt=5, neigh_orders=neigh_orders_2562, stride_t=3, padding_t=1, vertex_chunk_size=32,
                     residual=False),
        res_block_3D(16, 16, kt=5, neigh_orders=neigh_orders_2562, stride_t=1, padding_t=2, vertex_chunk_size=32),
        pool_layer_batch_3D(neigh_orders_2562, 'mean'),
        res_block_3D(16, 16, kt=5, neigh_orders=neigh_orders_642, stride_t=1, padding_t=2, vertex_chunk_size=32),
    )
    data = torch.randn(2,200,40962)
    out = model(data)
    print(out.shape)
    out = pool(out)
    print(out.shape)
    out = model1(out)
    print(out.shape)
    # out = model2(out)
    # print(out.shape)
    out = model4(data)
    print(out.shape)
