# Point Cloud

## PointNet

来自这篇[知乎](https://zhuanlan.zhihu.com/p/336496973)
![PointNet](./pics/pointnet.jpg "architecture")

1. 变换矩阵（**T-Net**）
![T-Net](pics/T-Net.jpg)
为了保证输入点云的**不变性**，作者在进行特征提取前先对点云数据进行**对齐操作**：因为点云从各个方向上观测虽然不同，但其表示的是同一个物体，因此可以对齐到一个空间上（也就是input transform）。  
**对齐操作**是通过训练一个小型的网络（也就是上图中的T-Net）来得到**转换矩阵**，再将转换矩阵与输入相乘实现的。

```python
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]  # shape (batch_size,3,point_nums)
        x = F.relu(self.bn1(self.conv1(x)))  # shape (batch_size,64,point_nums)
        x = F.relu(self.bn2(self.conv2(x)))  # shape (batch_size,128,point_nums)
        x = F.relu(self.bn3(self.conv3(x)))  # shape (batch_size,1024,point_nums)
        x = torch.max(x, 2, keepdim=True)[0]  # shape (batch_size,1024,1)
        x = x.view(-1, 1024) # shape (batch_size,1024)

        x = F.relu(self.bn4(self.fc1(x)))  # shape (batch_size,512)
        x = F.relu(self.bn5(self.fc2(x)))  # shape (batch_size,256)
        x = self.fc3(x)  # shape (batch_size,9)

        """
        最终的 3*3变换矩阵 是要与 n*3的点云矩阵 相乘来实现变换的，而实际上其定义变换乘法是：
        输入 矩阵乘 变换矩阵 + 输入(添加恒等变换)，这个恒等变换是通过在中间矩阵添加iden实现的
        """ 
        # iden表示一个对角阵(1填充的)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # shape (batch_size,9)
        if x.is_cuda:
            iden = iden.cuda()
        # that's the same thing as adding a diagonal matrix(full 1)
        x = x + iden  # iden means that add the input-self
        x = x.view(-1, 3, 3) # shape (batch_size,3,3)
        return x  # 得到的x是变换矩阵3*3
```

和上面的input transform矩阵的获取方式类似，feature transform的`64*64`矩阵获取代码实现如下：

```python
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
```

其在**分割**任务上的还原过程：
![PointNet](./pics/pointnet.jpg "注意图片下半部分（分割）")
这个`n*1088`的张量由两部分组成，一个是**特征提取网络的输出**（大小为`n*64`）,另一个是通过maxpooling后的global feature（大小为1024），在进行两者融合的时候，对global feature进行了广播，那么`64+1024`就是`1088`了。为什么要这么做呢？:答案就是作者想要融合**点的特征信息**（来自特征提取网络的输出）与**全局特征**（来自global feature）。

> **缺点**：一直在通过mlp进行单个点内的信息交互，最后通过Max Pooling只进行了一次全局信息的交互，缺少中间尺度的信息交互(如CNN中的逐尺度下采样特征提取)，缺少局部特征信息。

👇

## PointNet++

> **Sampling Layer + Grouping Layer**  

### **最远点采样算法(FPS)**来实现从`N`个点中采样`N'`个点👉**Sampling Layer**

1. 随机选择一个点作为**初始点**作为**已选择采样点**
2. 计算**未选择采样点集**中每个点与**已选择采样点集**之间的距离distance，将**距离最大**的那个点加入已选择采样点集
3. 更新distance，一直循环迭代下去，直至获得了目标数量的采样点。

```python
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape  # (B, N, 3)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 初始化目标采样点(B, npoint)
    distance = torch.ones(B, N).to(device) * 1e10  # (B, N)
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # (B,)表示在batch中的 每一个样本 里随机初始化一个点作为基准(包含的是最远点的idx)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 等于在切片中直接写":"
    for i in range(npoint):
        centroids[:, i] = farthest  # (b, npoint)
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B, 3) -> (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N, 3) - (B, 1, 3) -> dist(B, N):选出来的点与所有点的最小二乘(包含的是当前点与所有点的距离)
        mask = dist < distance  # 你与上一个最远点的距离肯定是大于当前计算的所有的dist，因此上一个最远点处会被标为false，避免选点来回在两个最远点之间跳摆。
        distance[mask] = dist[mask]  # 更新distance为排除所选最远点之后的值
        farthest = torch.max(distance, -1)[1]
    return centroids
#  当B=4， N=5, npoint=3时的结果
>>
farthest0:                                  farthest1:
tensor([0, 3, 4, 2])                        tensor([1, 0, 1, 0])
dist:                                       dist:
tensor([[   0, 9821, 4853, 5709, 4358],     tensor([[9821,    0, 5082, 1856, 9011],
        [1787,  708, 1469,    0, 1217],             [   0, 3955, 3362, 1787, 1702],
        [7178, 9595, 8797, 4309,    0],             [4497,    0,  694, 1314, 9595],
        [3924,  801,    0, 1294,  152]])            [   0, 3589, 3924, 1986, 3948]])
mask:                                       mask:#可看到上一次选过的点处已经被标为了False
tensor([[True, True, True, True, True],     tensor([[False,  True, False,  True, False],
        [True, True, True, True, True],             [ True, False, False, False, False],
        [True, True, True, True, True],             [ True,  True,  True,  True, False],
        [True, True, True, True, True]])            [ True, False, False, False, False]])
distance:                                   distance:
tensor([[   0, 9821, 4853, 5709, 4358],     tensor([[   0,    0, 4853, 1856, 4358],
        [1787,  708, 1469,    0, 1217],             [   0,  708, 1469,    0, 1217],
        [7178, 9595, 8797, 4309,    0],             [4497,    0,  694, 1314,    0],
        [3924,  801,    0, 1294,  152]])            [   0,  801,    0, 1294,  152]])

farthest2:
tensor([2, 2, 0, 3])
dist:
tensor([[ 4853,  5082,     0,  6410, 12753],
        [ 3362,  3413,     0,  1469,  5022],
        [    0,  4497,  2379,  3681,  7178],
        [ 1986,   321,  1294,     0,  1126]])
mask:
tensor([[False, False,  True, False, False],
        [False, False,  True, False, False],
        [ True, False, False, False, False],
        [False,  True, False,  True, False]])
distance:
tensor([[   0,    0,    0, 1856, 4358],
        [   0,  708,    0,    0, 1217],
        [   0,    0,  694, 1314,    0],
        [   0,  321,    0,    0,  152]])
```

👇  
**如何将点集划分为不同的区域，并获取不同区域的局部特征？**

### Ball query(group策略)👉**Grouping Layer**

1. 预设**搜索区域**的半径`R`与**子区域**的点数`K`
2. 根据上面提取的`N'`确定centriods数量，以`N'`个点为球心，画半径为`R`的球体（叫做`query ball`，也就是搜索区域）。
3. 在每个以centriods的球心的球体内搜索离centriods最近的的点（按照距离从小到大排序，找到`K`个点）。如果`query ball`的点数量大于点数`K`，那么直接取前`K`个作为子区域；(**如果小于，那么直接对某个点重采样，凑够规模`K`**不是很理解如何重采样)

👇  

### 进行局部特征的提取 **Set Abstraction(SA)**

> 通过Sample layer和Grounping layer后，网络后面紧跟着一个pointnet来进行区域特征提取  
![pointnet_in_PN++](./pics/pointnet_in_PN++.jpg)  
作者将max pool用在子区域上，实现**区域**特征提取  
`Sample layer/Grounping layer/Pointnet`（三个合在一起叫做`set abstraction`）

一个`set abstraction`代码如下：

```python
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # (B, N, C)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, N, D)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)  # 采样覆盖所有点
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
```

> **缺点**：来源于sampling和grouping的在遇到**非均匀分布**的点云集合时：
>  
> `It is common that a point set comes with` **nonuniform density** `in` **different areas**  
> `Features learned in` **dense** `data may` **not generalize to** **sparsely** `sampled regions`
