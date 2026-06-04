import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import sys
sys.path.append('/ptycho_gui/')
from nsls2ptycho2.core.ptycho.utils import parse_config
from nsls2ptycho2.core.ptycho.recon_ptycho_gui import recon_thread

#param = parse_config('/nsls2/data2/hxn/legacy/users/2025Q1/Boyu_2025Q1/ptycho/recon_result/S334806/testlive/recon_data/334806_testlive.ptycho_root_8691_57.txt')
param = parse_config('./ptycho_config.txt')

if True:
    # param.live_recon_flag = True

    recon,rank = recon_thread(param)
    recon.setup()


    # diff_d = np.load('diff_d.npy')
    # point_info_d = np.load('point_info_d.npy')

    # cp.cuda.runtime.memcpy(recon.diff_d.data.ptr,diff_d.ctypes.data,diff_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)
    # cp.cuda.runtime.memcpy(recon.point_info_d.data.ptr,point_info_d.ctypes.data,point_info_d.nbytes,cp.cuda.runtime.memcpyHostToDevice)

    # recon.num_points_recon = 10000
    # recon.live_update_plan_last()

    # for it in range(50):
    #     recon.one_iter(it)
