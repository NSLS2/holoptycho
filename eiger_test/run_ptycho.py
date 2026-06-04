gpus = [0]
scannum = 257331

from nsls2ptycho.core.ptycho.utils import parse_config,save_config
from nsls2ptycho.core.ptycho_param import Param
# import numpy as np
import os

if __name__=='__main__':
    # basedir = os.path.dirname(os.path.realpath(__file__))+'/'
    # os.chdir(basedir)
    param = parse_config('ptycho_config',Param())
    param.working_directory = "/test_data/"
    param.scan_num = scannum
    param.gpus = gpus
    save_config('ptycho_config',param)

    print('mpirun -n %d python3.11 -m nsls2ptycho.core.ptycho.recon_ptycho_gui ptycho_config'%(len(param.gpus)))
    os.system('mpirun -n %d python3.11 -m nsls2ptycho.core.ptycho.recon_ptycho_gui ptycho_config'%(len(param.gpus)))
