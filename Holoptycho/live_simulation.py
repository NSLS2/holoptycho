import logging
from time import sleep
import time

import sys
import os
import h5py

import numpy as np
import cupy as cp
from numba import cuda

from hxntools.motor_info import motor_table


from ..ptycho.utils import parse_config
from ..ptycho.recon_ptycho_gui import recon_thread

from holoscan.core import Application, Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from .datasource import parse_args, EigerZmqRxOp, PositionRxOp, EigerDecompressOp
from .preprocess import ImageBatchOp, ImagePreprocessorOp, PointProcessorOp, ImageSendOp
from .liverecon_utils import parse_scan_header

class InitSimul(Operator):
    def __init__(self, *args, param,batchsize,min_points, **kwargs):
        super().__init__(*args,**kwargs)

        self.batchsize = batchsize
        self.min_points = min_points
        self.angle_correction_flag = True

        self.param = param
        self.scan_num = param.scan_num
        self.working_dir = param.working_directory
        self.h5_file = self.working_dir+'/scan_'+str(self.scan_num)+'.h5'

        self.h5_header = h5py.File(self.h5_file,'r',locking=False)
        self.ic = np.array(self.h5_header['ic'])
        self.ic = self.ic/np.mean(self.ic)
        if 'raw_data' in self.h5_header.keys() and self.h5_header['raw_data/flag'][()]:
            self.rawdata_filename = self.h5_header['raw_data/filename'][()]
            self.roi = np.array(self.h5_header['raw_data/roi'])
            self.badpixels = np.array(self.h5_header['raw_data/badpixels'])
            self.nx = self.roi[0,1] - self.roi[0,0]
            self.ny = self.roi[1,1] - self.roi[1,0]
            self.h5_raw = h5py.File(self.rawdata_filename[0],'r',locking=False)
            self.rawdata = self.h5_raw['entry/data/data']
        else:
            self.h5_raw = None
            self.rawdata  = self.h5_header['diffamp']
            self.nx,self.ny,_ = self.rawdata.shape
        self.nz = self.h5_header['points'].shape[1]
        self.points = np.array(self.h5_header['points'][:]) # scan grid info
        self.points_simulate = np.zeros((2,self.points.shape[1]*10))
        self.points_simulate[0] = np.repeat(self.points[0],10)
        self.points_simulate[1] = np.repeat(self.points[1],10)
        
        self.nz = self.nz - self.nz%self.batchsize
        self.x_num = self.param.x_range // self.param.dr_x

        if self.angle_correction_flag:
            if np.abs(self.param.angle) <= 45.:
                self.param.x_range *= np.abs(np.cos(self.param.angle*np.pi/180.))
            else:
                self.param.x_range *= np.abs(np.sin(self.param.angle*np.pi/180.))

        self.counter = 0
        self.point_datapack_counter = 0



    def setup(self,spec):
        spec.output("flush_image_send").condition(ConditionType.NONE)
        spec.output("flush_pos_proc").condition(ConditionType.NONE)
        spec.output("flush_pty").condition(ConditionType.NONE)

        spec.output("diff_amp").condition(ConditionType.NONE)
        spec.output("image_indices").condition(ConditionType.NONE)

        spec.output("pointRx_out").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):
        if self.counter == 0:
            # flush to begin
            op_output.emit(True,'flush_image_send')
            op_output.emit((self.param.x_range,self.param.y_range,1,1,self.x_num*2,self.param.angle,False),'flush_pos_proc')
            op_output.emit((self.scan_num,self.param.x_range,self.param.y_range,np.maximum(self.x_num*2,self.min_points),self.nz),'flush_pty')

        if self.counter < self.nz:
            i = self.counter
            if self.h5_raw is not None:
                detmap = np.array(self.rawdata[i:i+self.batchsize])

                for bd in self.badpixels.T:
                    x = int(bd[0])
                    y = int(bd[1])

                    # Skip bad pixels outside the roi
                    if x>=self.roi[0,0] and x<self.roi[0,1] and y>=self.roi[1,0] and y<self.roi[1,1]:
                        for iz in range(self.batchsize):
                            detmap[iz,x,y] = np.median(detmap[iz,x-1:x+2,y-1:y+2])
                    
                detmap = detmap[:,self.roi[0,0]:self.roi[0,1],self.roi[1,0]:self.roi[1,1]]
                detmap = np.rot90(detmap,axes=(2,1))
                detmap = np.fft.fftshift(detmap,axes=(1,2))
                diff_l = np.sqrt(detmap,dtype = np.float32,order='C')
            else:
                diff_l = np.array(self.rawdata[i:i+self.batchsize],dtype = np.float32,order='C')

            op_output.emit(diff_l, "diff_amp")
            op_output.emit(np.arange(i,i+self.batchsize), "image_indices")
            op_output.emit((self.point_datapack_counter,self.points_simulate[:,i*10:(i+self.batchsize)*10]), "pointRx_out")

            self.counter += self.batchsize
            self.point_datapack_counter += 1
            time.sleep(self.batchsize / 1000)
        else:
            self.h5_header.close()
            if self.h5_raw:
                self.h5_raw.close()
            self.stop_execution()
            
        # self.stop_execution()
