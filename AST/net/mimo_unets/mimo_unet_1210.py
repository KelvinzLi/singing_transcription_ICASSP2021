import torch.nn as nn
import torch
import torch.nn.functional as F

from net.ConvModules import ConvLayer2d, ConcatConnect, ResConnect

class DimMapper(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = torch.permute(x, (0, 1, 3, 2))
        x = torch.flatten(x, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return torch.permute(x, self.shape)

class Unet(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super().__init__()
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class
        
        m = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvLayer2d(128, 256, 5, 1, 2),
            ResConnect(ConvLayer2d(256, 256, 5, 1, 2, activation=nn.Identity())),
            ResConnect(ConvLayer2d(256, 256, 5, 1, 2, activation=nn.Identity())),
            nn.Upsample(scale_factor=2),
        )
        m = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvLayer2d(64, 128, 3, 1, 1),
            ConvLayer2d(128, 128, 3, 1, 1),
            
            ConcatConnect(m),
            
            ConvLayer2d(256+128, 128, 5, 1, 2),
            nn.Upsample(scale_factor=2),
        )
        m = nn.Sequential(
            ConvLayer2d(32, 64, 3, 1, 1),
            ConvLayer2d(64, 64, 3, 1, 1),
            ConcatConnect(m),
            ConvLayer2d(64+128, 64, 5, 1, 2),
        )
        m = nn.Sequential(
            ConvLayer2d(1, 32, 7, 1, 3),
            m,
            DimMapper(),
            nn.Conv1d(64 * 168, 2+pitch_class+pitch_octave+2, 1, 1, 0), # Taking up a lot of memory now!
        )
        
        self.m = m

        
    def forward(self, x):
        out = self.m(x)
        # print(out.shape)
        # [batch, output_size]

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]

        pitch_out = out[:, 2:]
        
        pitch_octave_logits = pitch_out[:, 0:self.pitch_octave+1]
        pitch_class_logits = pitch_out[:, self.pitch_octave+1:]

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits

if __name__ == '__main__':
    # from torchsummary import summary
    model = Unet().cuda()
    # summary(model, input_size=(1, 11, 168))
    print(sum(p.numel() for p in model.parameters()))