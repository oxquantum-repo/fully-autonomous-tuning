# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:01:25 2021

@author: Mathieudk
"""
from typing import (Optional, List, Sequence, Union, Tuple, Dict,
                    Any, Set, cast)

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from qcodes.dataset.data_export import (get_data_by_id, flatten_1D_data_for_plot,reshape_2D_data)
from qcodes import (load_by_run_spec)
from qcodes.dataset.plotting import  _set_data_axes_labels,_rescale_ticks_and_units
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

def dataset_to_numpy(dataset):
    alldata = get_data_by_id(dataset.run_id)
    
    x = []
    y = []
    z = []
    
    for i in range(len(alldata)):
        data = alldata[i][:]
            
        if len(data) == 2:
            x_temp = flatten_1D_data_for_plot(data[0]['data'])
            y_temp = flatten_1D_data_for_plot(data[1]['data'])
            
            x.append(x_temp)
            y.append(y_temp)
        
        if len(data) == 3:
            xpoints = flatten_1D_data_for_plot(data[0]['data'])
            ypoints = flatten_1D_data_for_plot(data[1]['data'])
            zpoints = flatten_1D_data_for_plot(data[2]['data'])
    
            x_temp,y_temp,z_temp = reshape_2D_data(xpoints,ypoints,zpoints)
            
            x.append(x_temp)
            y.append(y_temp)
            z.append(z_temp)
        
    return x,y,z
    
def datasetID_to_numpy(ID):
    alldata = get_data_by_id(ID)
    
    x = []
    y = []
    z = []
    
    for i in range(len(alldata)):
        data = alldata[i][:]
            
        if len(data) == 2:
            x_temp = flatten_1D_data_for_plot(data[0]['data'])
            y_temp = flatten_1D_data_for_plot(data[1]['data'])
            
            x.append(x_temp)
            y.append(y_temp)
        
        if len(data) == 3:
            xpoints = flatten_1D_data_for_plot(data[0]['data'])
            ypoints = flatten_1D_data_for_plot(data[1]['data'])
            zpoints = flatten_1D_data_for_plot(data[2]['data'])
    
            x_temp,y_temp,z_temp = reshape_2D_data(xpoints,ypoints,zpoints)
            
            x.append(x_temp)
            y.append(y_temp)
            z.append(z_temp)
        
    return x,y,z    


def plot2D_by_ID(ID,slice: Optional=0):
    alldata = get_data_by_id(ID)
    data = alldata[slice][:]
    
    xpoints = flatten_1D_data_for_plot(data[0]['data'])
    ypoints = flatten_1D_data_for_plot(data[1]['data'])
    zpoints = flatten_1D_data_for_plot(data[2]['data'])

    x_temp,y_temp,z_temp = reshape_2D_data(xpoints,ypoints,zpoints)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormesh = ax.pcolormesh(x_temp,y_temp,z_temp,cmap = "magma",shading = "nearest")
    colorbar = ax.figure.colorbar(colormesh, ax=ax)
    
    _set_data_axes_labels(ax, data, colorbar)
    _rescale_ticks_and_units(ax, data, colorbar)
    
    return fig,ax


def define_detuning(ID, slice: Optional=0):

    fig, ax = plot2D_by_ID(ID, slice)
    
    line, = line, = ax.plot([], [])
    detuning = det_help(fig, ax)
    plt.show()
    
    return detuning

class det_help:
    
    def __init__(self, fig, ax):
        
        self.perp_slope = None
        self.drawn = False
        self.xs = []
        self.ys = []

        self.fig = fig
        self.ax = ax
        line, = self.ax.plot([], [])  # empty line
        self.lines=[]

        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        
        cid1 = fig.canvas.mpl_connect('button_press_event',self.on_press)
        cid2 = fig.canvas.mpl_connect('key_press_event',self.move_line)
        cid3 = fig.canvas.mpl_connect('key_press_event',self.confirm_det)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        self.xlabel = ax.get_xlabel()
        self.ylabel = ax.get_ylabel()

    def on_press(self,event):
        if self.drawn==True:
            temp = list(self.ax.get_lines())
            for temp_line in temp[1:]:
                temp_line.remove()

        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.drawn = False
        if len(self.xs)==2:
            self.lines.append(self.ax.plot(self.xs,self.ys,'.-',color = 'k'))
            dx = self.xs[1] - self.xs[0]
            dy = self.ys[1] - self.ys[0]
            slope = dy/dx
            self.perp_slope = -1/slope
            x_perp = np.array(self.ax.get_xlim())
            self.x_perp = 3*x_perp
            self.zdl_slice = 0
            self.offset_perp = np.linspace(self.ys[0]-self.perp_slope*self.xs[0],self.ys[1]-self.perp_slope*self.xs[1],26)
            self.lines.append(self.ax.plot(self.x_perp,self.perp_slope*self.x_perp+self.offset_perp[self.zdl_slice],color = 'C0'))
            self.xs[:] = []
            self.ys[:] = []

            self.drawn = True

        fig.canvas.draw_idle()

    def move_line(self,event):
        if self.drawn == False: return
        if event.key != 'left' and event.key != 'right': return
        temp = list(self.ax.get_lines())
        temp_line = temp[-1]
        temp_line.remove()

        if event.key == 'right':
            zdl_slice_new = np.mod(self.zdl_slice+1,26)
            self.zdl_slice = zdl_slice_new
        if event.key == 'left':
            zdl_slice_new = np.mod(self.zdl_slice-1,26)
            self.zdl_slice = zdl_slice_new

        self.lines.append(self.ax.plot(self.x_perp,self.perp_slope*self.x_perp+self.offset_perp[self.zdl_slice],color = 'C0'))
        fig.canvas.draw_idle()
    
    def disconnect(self):
        'disconnect all the stored connection ids'
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        fig.canvas.mpl_disconnect(cid3)   

    def confirm_det(self,event):
        if self.drawn == False: return
        if event.key != 'enter': return
        self.slope = self.perp_slope
        self.offset = self.offset_perp[self.zdl_slice]
        
        filename = "detunig_axis.txt"
        if os.path.exists(filename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        detfile = open(filename,append_write)
        now = datetime.now()
        
        now_str = now.strftime("%m/%d/%Y, %H:%M:%S")
        slope_str = str(np.round(self.slope,5))
        offset_str = str(np.round(self.offset,5))
        
        total_str = now_str + ", " + self.xlabel + ", " + self.ylabel + ", " + slope_str + ", " + offset_str
        
        detfile.write(total_str + '\n')
        
        self.disconnect()


     