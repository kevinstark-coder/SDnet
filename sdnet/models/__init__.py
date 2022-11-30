from .sdnet import *
from .sdnet2 import *
from .sdnet3 import *
from .weight_init import *
from .sdnet_cycle_zout import *
from .sdnet_zjy import *

import sys

def get_model(name, params):
    if name == 'sdnet':
        return SDNet(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet2':
        return SDNet2(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet3':
        return SDNet3(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet_cycle_zout':
        return SDNet_cycle_zout(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])
    elif name == 'sdnet_zjy':
        return SDNet_zjy(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['anatomy_out_channels'], params['num_mask_channels'])

    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)