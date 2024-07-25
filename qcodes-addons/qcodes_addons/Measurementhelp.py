import sys
from typing import Optional

import numpy as np
from qcodes import config
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import _BaseParameter
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from qcodes.utils.dataset.doNd import (
    _catch_interrupts,
    _handle_plotting,
    process_params_meas,
)
from qcodes_addons.NIDAQ import DAQVoltageMultiChannel
from tqdm.auto import tqdm


def do1d_time(iterations,
              *param_meas,
              write_period: Optional[float] = None,   
              do_plot: Optional[bool] = None,
              use_threads: bool = False,              
              show_progress: Optional[None] = None):
    
    if do_plot is None:
        do_plot = config.dataset.dond_plot
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress
    
    param_time = ElapsedTimeParameter('time')
    meas = Measurement()
    
    if write_period is not None:
        meas.write_period = write_period
    
    meas.register_parameter(param_time)
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints = (param_time,))
    
   
        
    param_time.reset_clock()
    for _ in tqdm(range(iterations), disable=not show_progress):
        datasaver.add_result((param_time, param_time()),
                                *process_params_meas(param_meas, use_threads=use_threads))
    
    dataset = datasaver.dataset
      
    return _handle_plotting(dataset, do_plot, interrupted())
        
        
def do2d_time(param_set, 
              start, 
              stop, 
              num_steps, 
              delay, 
              iterations, 
              *param_meas,
              write_period: Optional[float] = None,              
              do_plot: Optional[bool] = None,
              use_threads: bool = False,
              show_progress: Optional[None] = None):              
                  
    if do_plot is None:
        do_plot = config.dataset.dond_plot
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress
        
    param_time = ElapsedTimeParameter('time')
    meas = Measurement()
    
    if write_period is not None:
        meas.write_period = write_period
    
    meas.register_parameter(param_time)
    meas.register_parameter(param_set)
    
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=(param_time, param_set,))
    param_set.post_delay = delay
    
    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        setpoints = np.linspace(start, stop, num_steps)
        
        # flush to prevent unflushed print's to visually interrupt tqdm bar
        # updates
        sys.stdout.flush()
        sys.stderr.flush()
        
        param_time.reset_clock()
        for _ in tqdm(range(iterations), disable=not show_progress):
            now = param_time()
            for set_v in tqdm(setpoints, disable=not show_progress, leave=False):
                param_set.set(set_v)
                datasaver.add_result((param_time, now), 
                                     (param_set, set_v), 
                                     *process_params_meas(param_meas, use_threads=use_threads))
                                    
        dataset = datasaver.dataset
    
    return _handle_plotting(dataset, do_plot, interrupted())


def do1d_realtime(param, stoptime,
                  write_period: Optional[float] = None,
                  do_plot: Optional[bool] = None, 
                  show_progress: Optional[None] = None):
    
    multichannel = False
    meas = Measurement()
    
    if write_period is not None:
        meas.write_period = write_period
        
    
    if type(param) is DAQVoltageMultiChannel:
        params = []
        multichannel = True
        for ch in param.chs:
            params.append(ch.volt_time)
        data = [0]*len(params)
        # switch DAQ from finite to continuous sampling
        param.chs[0].instrument.switch_sample_mode()
        numpoints = int(stoptime/param.chs[0].instrument.update_interval)
        for p in params:
            meas.register_parameter(p)
    else:
        meas.register_parameter(param)
        # switch DAQ from finite to continuous sampling
        param.instrument.instrument.switch_sample_mode() 
        numpoints = int(stoptime/param.instrument.instrument.update_interval)
        
  
    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        # flush to prevent unflushed print's to visually interrupt tqdm bar
        # updates
        sys.stdout.flush()
        sys.stderr.flush()
        for _ in tqdm(range(numpoints), disable=not show_progress):
            data = param.get()
            if multichannel is False:
                timedata = param.instrument.instrument.time_axis()
                datasaver.add_result((param, data),(param.instrument.instrument.time_axis, timedata))
            else:
                result = list(zip(params,data))
                timedata = param.chs[0].instrument.time_axis()
                datasaver.add_result(*result,
                                    (param.chs[0].instrument.time_axis,timedata))                       
        dataset = datasaver.dataset
    
    # switch DAQ back from continuous to finite sampling
    if multichannel == False:
        param.instrument.instrument.switch_sample_mode()
    else:
        param.chs[0].instrument.switch_sample_mode()
    
    return _handle_plotting(dataset, do_plot, interrupted())
                  
        
def do1d_break(param_set, 
               start,
               stop,
               num_points,
               delay,
               *param_meas,
               limits,
               write_period: Optional[float] = None,              
               do_plot: Optional[bool] = None,
               use_threads: bool = False,
               show_progress: Optional[None] = None):

    if do_plot is None:
        do_plot = config.dataset.dond_plot
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress    

    meas = Measurement()
    
    if write_period is not None:
        meas.write_period = write_period
    
    meas.register_parameter(param_set)
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=(param_set,))
    param_set.post_delay = delay
    
   
    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        setpoints = np.linspace(start, stop, num_points)
        
        # flush to prevent unflushed print's to visually interrupt tqdm bar
        # updates
        sys.stdout.flush()
        sys.stderr.flush()
        
        for set_point in tqdm(setpoints, disable=not show_progress):
            param_set.set(set_point)
            
            data = process_params_meas(param_meas, use_threads=use_threads)            
            datasaver.add_result((param_set, set_point), *data)
            
            meas_vals = []
            for i in range(len(param_meas)):
                for elem in np.atleast_1d(data[i][1]):
                    meas_vals.append(np.abs(elem))
            if any(np.greater(meas_vals, limits)):
                print('MEASUREMENT STOPPED: protection limits have been exceeded')
                break
                                    
        dataset = datasaver.dataset
    
    return _handle_plotting(dataset, do_plot, interrupted())
      
  
def do2d_break(param_set1, 
               start1,
               stop1,
               num_points1,
               delay1,
               param_set2,
               start2,
               stop2,
               num_points2,
               delay2,
               *param_meas,
               limits,
               write_period: Optional[float] = None,              
               do_plot: Optional[bool] = None,
               use_threads: bool = False,
               show_progress: Optional[None] = None):
              
    if do_plot is None:
        do_plot = config.dataset.dond_plot
    if show_progress is None:
        show_progress = config.dataset.dond_show_progress    
        
    meas = Measurement()
    
    if write_period is not None:
        meas.write_period = write_period
    
    meas.register_parameter(param_set1)
    meas.register_parameter(param_set2)
    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            meas.register_parameter(parameter,
                                    setpoints=(param_set1,param_set2))
    param_set1.post_delay = delay1
    param_set2.post_delay = delay1
        
    with _catch_keyboard_interrupts() as interrupted, meas.run() as datasaver:
        setpoints1 = np.linspace(start1, stop1, num_points1)
        setpoints2 = np.linspace(start2, stop2, num_points2)
        
        breaker = False
        for set_point1 in tqdm(setpoints1, disable=not show_progress):
            param_set1.set(set_point1)
            
            # flush to prevent unflushed print's to visually interrupt tqdm bar
            # updates
            sys.stdout.flush()
            sys.stderr.flush()
            
            for set_point2 in tqdm(setpoints2,
                                   disable=not show_progress,
                                   leave=False):
                param_set2.set(set_point2)
                
                data = process_params_meas(param_meas, use_threads=use_threads)            
                datasaver.add_result((param_set1, set_point1),
                                     (param_set2, set_point2),
                                     *data)            
                meas_vals = []
                for i in range(len(param_meas)):
                    for elem in np.atleast_1d(data[i][1]):
                        meas_vals.append(np.abs(elem))
                if any(np.greater(meas_vals, limits)):
                    breaker = True
                    print('MEASUREMENT STOPPED: protection limits have been exceeded')
                    break
                    
            if breaker == True:
                break
                                    
        dataset = datasaver.dataset
    
    return _handle_plotting(dataset, do_plot, interrupted())