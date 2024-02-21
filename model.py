   

class ResNeXtBottleneck(nn.Module):

    def __init__(self, in_channels,  pool_stride, cardinality):
        '''
        Constructor

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: convolutional stride; replaces pooliong layer
            cardinality: number of convolutional groups
            base_width: base number of channels in each group
            widen_factor: factor to reduce the input dimensionality before convolution
        '''
        super(ResNeXtBottleneck, self).__init__()

        D =4*cardinality
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=pool_stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(self.bn_reduce.forward(bottleneck),negative_slope=0.2, inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(self.bn.forward(bottleneck),negative_slope=0.2, inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        return F.leaky_relu(x + bottleneck,negative_slope=0.2, inplace=True)

   


class ResNeXt(nn.Module):

    def __init__(self, cardinality,num_blocks,scale_factor):
        
        upsample_block_num = int(math.log(scale_factor, 2))

        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.num_blocks = num_blocks
        self.n_heads = 1


        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = self.make_layer2( 64, self.num_blocks , stride=1)
        self.block3 = self.make_layer2( 64,  self.num_blocks, stride=1)
        self.block4 = self.make_layer2( 64,  self.num_blocks, stride=1)
        self.block5 = self.make_layer2( 64,  self.num_blocks, stride=1)
        self.block6 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        block7 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block7.append(spectral_norm(nn.Conv2d(64, 3, kernel_size=9, padding=4)))
        self.block7 = nn.Sequential(*block7)
        

                               
       
    
    def make_layer2(self, channels, blocks, stride=1):
        # Création des couches parallèles de convolutions
        layers = []
        self.channels = channels
        for i in range( blocks):
            layers.append(ResNeXtBottleneck(self.channels, stride, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)  
        
        
        
        block5 = self.block5(block4)
         
        
        
        
        
        block6 = self.block6(block5)
        block7 = self.block7(block1 + block6)

        return (torch.tanh(block7) + 1) / 2




class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator_UNet, self).__init__()
        self.n_heads = 1
        norm = spectral_norm

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(ndf, 1, 3, 1, 1)
        print('using the UNet discriminator')
        
        self.attn1 = nn.MultiheadAttention(ndf*8, num_heads=self.n_heads)
        self.alpha1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.attn2 = nn.MultiheadAttention(ndf*4, num_heads=self.n_heads)
        self.alpha2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        #### Attention 1 ####
        identity1 = x3
        B, C, H, W = x3.shape
        q_1 = x3.view(H * W, B, C)
        k_1 = x3.view(H * W, B, C)
        v_1 = x3.view(H * W, B, C)
        x3, attn_map_1 = self.attn1(q_1, k_1, v_1)
        x3 = x3.view(B, C, H, W)

        # Residual connection
        x3 = identity1 + self.alpha1 * x3
        
        
        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        
        #### Attention 2 ####
        identity2 = x4
        B, C, H, W = x4.shape
        q_2 = x4.view(H * W, B, C)
        k_2 = x4.view(H * W, B, C)
        v_2 = x4.view(H * W, B, C)
        x4, attn_map_2 = self.attn2(q_2, k_2, v_2)
        x4 = x4.view(B, C, H, W)

        # Residual connection
        x4 = identity2 + self.alpha2 * x4
        
        
        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return torch.sigmoid(out)
    
