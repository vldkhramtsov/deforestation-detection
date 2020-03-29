import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad3d, ReplicationPad3d


class Unet3D(nn.Module):
    """EF segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(Unet3D, self).__init__()

        self.input_nbr = input_nbr

        self.conv11 = nn.Conv3d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(16)
        self.do11 = nn.Dropout3d(p=0.2)
        self.conv12 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(16)
        self.do12 = nn.Dropout3d(p=0.2)

        self.conv21 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm3d(32)
        self.do21 = nn.Dropout3d(p=0.2)
        self.conv22 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm3d(32)
        self.do22 = nn.Dropout3d(p=0.2)

        self.conv31 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm3d(64)
        self.do31 = nn.Dropout3d(p=0.2)
        self.conv32 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm3d(64)
        self.do32 = nn.Dropout3d(p=0.2)
        self.conv33 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm3d(64)
        self.do33 = nn.Dropout3d(p=0.2)

        self.conv41 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm3d(128)
        self.do41 = nn.Dropout3d(p=0.2)
        self.conv42 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm3d(128)
        self.do42 = nn.Dropout3d(p=0.2)
        self.conv43 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm3d(128)
        self.do43 = nn.Dropout3d(p=0.2)


        self.upconv4 = nn.ConvTranspose3d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose3d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm3d(128)
        self.do43d = nn.Dropout3d(p=0.2)
        self.conv42d = nn.ConvTranspose3d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm3d(128)
        self.do42d = nn.Dropout3d(p=0.2)
        self.conv41d = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm3d(64)
        self.do41d = nn.Dropout3d(p=0.2)

        self.upconv3 = nn.ConvTranspose3d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm3d(64)
        self.do33d = nn.Dropout3d(p=0.2)
        self.conv32d = nn.ConvTranspose3d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm3d(64)
        self.do32d = nn.Dropout3d(p=0.2)
        self.conv31d = nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm3d(32)
        self.do31d = nn.Dropout3d(p=0.2)

        self.upconv2 = nn.ConvTranspose3d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm3d(32)
        self.do22d = nn.Dropout3d(p=0.2)
        self.conv21d = nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm3d(16)
        self.do21d = nn.Dropout3d(p=0.2)

        self.upconv1 = nn.ConvTranspose3d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm3d(16)
        self.do12d = nn.Dropout3d(p=0.2)
        self.conv11d = nn.ConvTranspose3d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Forward method."""
        xs = []#x[0].unsqueeze(2)
        for tens_x in x:
            xs.append(tens_x.unsqueeze(2))
        x = torch.cat(xs, 2)
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool3d(x12, kernel_size=(1,2,2), stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool3d(x22, kernel_size=(1,2,2), stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool3d(x33, kernel_size=(1,2,2), stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool3d(x43, kernel_size=(1,2,2), stride=2)


        # Stage 4d
        x4d = self.upconv4(x4p)
        
        pad4 = ReplicationPad3d((0, x43.size(4) - x4d.size(4), 0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad3d((0, x33.size(4) - x3d.size(4), 0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad3d((0, x22.size(4) - x2d.size(4), 0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad3d((0, x12.size(4) - x1d.size(4), 0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        final = F.max_pool3d(x11d, kernel_size=(x11d.shape[2],1,1), stride=1)
        
        return final.squeeze().unsqueeze(1)
