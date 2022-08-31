import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
from De_Occlusion_Attention_Module import DOAM
import os
from PIL import Image
import cv2


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        a1) conv2d for class conf scores
        a2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,mode='cuda',type='ssd',ft_module=None,pyramid_ext=None):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        if type!='ssd':
            self.ft_module = nn.ModuleList(ft_module)
            self.pyramid_ext = nn.ModuleList(pyramid_ext)
        else:
            self.ft_module = ft_module
            self.pyramid_ext = ft_module

        self.loc = nn.ModuleList(head[0])
        self._conf = nn.ModuleList(head[1])

        self.edge_conv2d = DOAM(mode=mode)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
        if phase == 'onnx':
            self.softmax = nn.Softmax(dim=-1)
            # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    a1: confidence layers, Shape: [batch*num_priors,num_classes]
                    a2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [a2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        transformed_features = list()



        # apply vgg up to conv4_3 relu
        x = self.edge_conv2d(x)
        for k in range(23):
            #if(k==0):
                #img = x.int().cpu().squeeze().permute(1,2,0).detach().numpy()
                #cv2.imwrite('edge_s.jpg',img)
            #    x = self.edge_conv2d(x)
                #rgb_im = rgb_im.int().cpu().squeeze().permute(1,2,0).detach().numpy()
                #cv2.imwrite('rgb_im.jpg', rgb_im)
                #for i in range(6):
                #    im = Image.fromarray(edge_detect[i]*255).convert('L')
                #    im.save(str(i)+'edge.jpg')
                #x = self.edge_conv2d.edge_conv2d(x)
            #else:
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)


        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)


        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        #add fssd transformed_features
        if type!='ssd':
            assert len(self.ft_module) == len(sources)
            for k, v in enumerate(self.ft_module):
                transformed_features.append(v(sources[k]))
            x = torch.cat(transformed_features, 1)
            sources = list()
            for k, v in enumerate(self.pyramid_ext):
                x = v(x)
                sources.append(x)

        # apply multibox head to source layers
        #xfreq = [0,0,0,0,0]

        for (x, l, c) in zip(sources, self.loc, self._conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            #print(l(x).permute(0, 2, 3, 1).contiguous())
            #print(c)
        #print(loc)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #print(loc)
        #print("loc" + str(loc.size()))
        #print(len(sources))
        #for t in sources:
        #    print(t.size()[2])
        #    print(t.size(2))
        #print(xfreq)
        #a = 3;
        #print(self.priors.type(type(x.data)))
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data)),                  # default boxes

            )
        elif self.phase == 'onnx':
            output = (
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),
                self.priors.type(type(x.data)),
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        #print(output)
        return output

    def load_weights(self, base_file, isStrict = True):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')



            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage), strict = isStrict)

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    # '512': [],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 256, 256, 256,256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x=F.interpolate(input=x,size=(self.up_size, self.up_size), mode='bilinear')
        return x

def pyramid_feature_extractor(size):

    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers

def build_ssd(phase, size=300, num_classes=21,mode=None,type="ssd"):
    if phase != "test" and phase != "train" and phase != "onnx":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size == 300 or size == 512:
        pass
    else:
        raise("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300/SSD512 (size=300/512) is supported!")

    if(phase == 'train'):
        base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)
    elif(phase == 'test'):
        base_, extras_, head_ = multibox(vgg(base[str(size)], 4),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)
    elif(phase == 'onnx'):
        base_, extras_, head_ = multibox(vgg(base[str(size)], 4),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)

    if type =="ssd":
        layers=None
    else:
        if size == 300:
            up_size = 38
        elif size == 512:
            up_size = 64

        layers = []
        # conv4_3
        layers += [BasicConv(vgg[24].out_channels, 256, kernel_size=1, padding=0)]
        # fc_7
        layers += [BasicConv(vgg[-2].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]
        layers += [BasicConv(extras_[-1].out_channels, 256, kernel_size=1, padding=0, up_size=up_size)]

        pyramid_ext = pyramid_feature_extractor(size)


    return SSD(phase, size, base_, extras_, head_, num_classes,mode,type=type,ft_module=layers,pyramid_ext=pyramid_ext)
