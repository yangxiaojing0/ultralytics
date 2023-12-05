import torch.nn as nn
import torch

fusion_layer = [15, 18, 21]


class flow_adj(nn.Module):
    def __init__(self):
        super(flow_adj, self).__init__()

        # 残差+SE模块
        self.res_se = nn.Sequential(bottleneck_IR_SE(3, 16, 2),
                                    bottleneck_IR_SE(16, 32, 2),
                                    bottleneck_IR_SE(32, 64, 2),
                                    bottleneck_IR_SE(64, 128, 2),
                                    bottleneck_IR_SE(128, 256, 2))

        # BN层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.saved = [2, 3, 4]

        # p指针生成层
        self.p1 = Pointer(64, 16)
        self.p2 = Pointer(128, 16)
        self.p3 = Pointer(256, 16)

    def flow_3ds(self, flow):
        """
        前向传播，拿到三张光流特征图
        光流图依次经过残差+SE模块
        flow 需为[bs,3,640,640]
        """
        f_y = []

        # 光流图依次经过残差+SE模块
        for i, b_s in enumerate(self.res_se):
            flow = b_s(flow)
            f_y.append(flow if i in self.saved else None)

        return f_y

    def fusion(self, x, m, flow_tp):
        '''
        加权相加
        m:model
        x:原图特征图
        i：层数的名称，yolo原作者是以int命名的
        flow_tp：光流特征图
        '''
        if m.i == 15:
            p_g = self.p1(x, flow_tp[2])
            x = self.bn1(torch.add(p_g * x, (1 - p_g) * flow_tp[2]))
            # print('17--->', x.shape)
            
        if m.i == 18:
            p_g = self.p2(x, flow_tp[3])
            x = self.bn2(torch.add(p_g * x, (1 - p_g) * flow_tp[3]))
            # print('20--->', x.shape)

        if m.i == 21:
            p_g = self.p3(x, flow_tp[4])
            x = self.bn3(torch.add(p_g * x, (1 - p_g) * flow_tp[4]))
            # print('23--->', x.shape)

        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x  # [1,128,320,320]
        x = self.avg_pool(x)  # [1,128,320,320]---> [1,128,1,1]
        x = self.fc1(x)  # [1,128,1,1]--->[1,128/16,1,1]
        x = self.relu(x)
        x = self.fc2(x)  # [1,128/16,1,1]--->[1,128,1,1]
        x = self.sigmoid(x)
        return module_input * x


# 瓶颈模块添加SE模块
class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        # 短连接部分
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        # 残差部分加入se模块
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), stride, 1, bias=False),
            nn.PReLU(depth),
            SEModule(depth, 16)
        )

    # 前向传播
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Pointer(nn.Module):
    def __init__(self, in_channel, r):
        super(Pointer, self).__init__()
        # avg层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_rgb_1 = nn.Conv2d(in_channel, in_channel // r, (1, 1), padding=0)
        self.conv_rgb = nn.Conv2d(in_channel // r, in_channel, (3, 3), padding=1)
        self.conv_flow_1 = nn.Conv2d(in_channel, in_channel // r, (1, 1), padding=0)
        self.conv_flow = nn.Conv2d(in_channel // r, in_channel, (3, 3), padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_channel * 2, 1)

    def forward(self, rgb_fp, flow_fp):
        # 1.对两个fp进行通道拼接
        flow_output = self.conv_flow(self.relu(self.conv_flow_1(flow_fp)))

        rgb_output = self.conv_rgb(self.relu(self.conv_rgb_1(rgb_fp)))
        concat_fp = torch.concat([rgb_output, flow_output], dim=1)
        
        # concat_fp = torch.concat([rgb_fp, .flow_fp], dim=1)

        # 2. 使用avg_pool和全联接层进行纬度调整
        output = self.fc(self.avg_pool(concat_fp).reshape(-1, concat_fp.shape[1]))

        # 3. 使用sigmoid得到指针
        pointer = torch.sigmoid(output).reshape(-1, output.shape[1], 1, 1)

        return pointer


if __name__ == '__main__':
    """
    17---> torch.Size([1, 128, 48, 80])
    20---> torch.Size([1, 256, 24, 40])
    23---> torch.Size([1, 512, 12, 20])
    """
    dummy_f_17 = torch.randn([1, 128, 80, 80])
    dummy_f_20 = torch.randn([1, 256, 40, 40])
    dummy_f_23 = torch.randn([1, 512, 20, 20])
    x_17 = torch.randn([1, 128, 80, 80])
    x_20 = torch.randn([1, 256, 40, 40])
    x_23 = torch.randn([1, 512, 20, 20])
    conv = nn.Conv2d(128, 128, (3, 3), padding=1)
    print(conv(dummy_f_17).shape)
    # p = torch.randn([2, 1]).reshape(-1, 1, 1, 1)
    # xx = torch.randn([2, 128, 30, 30])
    # xx * p
    pointer = Pointer(128, 128)
    pg = pointer(dummy_f_17, x_17)
    print(pg.shape)
