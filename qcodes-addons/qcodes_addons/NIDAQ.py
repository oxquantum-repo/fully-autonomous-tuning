import queue
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import nidaqmx
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import (
    ChannelList,
    InstrumentChannel,
    MultiChannelInstrumentParameter,
)
from qcodes.instrument.parameter import Parameter, ParameterWithSetpoints
from qcodes.utils.validators import Arrays
from nidaqmx import stream_readers
from nidaqmx.constants import AcquisitionType


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_raw(self):
        num = self.root_instrument.num_of_samps_to_read_per_chan
        
        start = self.root_instrument.cnt_cont_mode*num
        start = start/self.root_instrument.rate # convert to time in s
        
        stop = (self.root_instrument.cnt_cont_mode+1)*(num-1)+self.root_instrument.cnt_cont_mode
        stop = stop/self.root_instrument.rate # convert to time in s
        
        setpts = np.linspace(start, stop, num)
        setpts = np.reshape(setpts, (num,1,))
        return setpts

    
class DAQVoltageMultiChannel(MultiChannelInstrumentParameter):  
    def __init__(self,
                 chs: Sequence[InstrumentChannel],
                 param_name: str,
                 *args: Any, 
                 **kwargs: Any) -> None:
        super().__init__(chs, param_name, *args, **kwargs)
        self.chs = chs
        self._param_name = param_name
        self.nchs = len(self.chs)
        self.lin = np.linspace(0,self.nchs-1,self.nchs)
        self.chnums = []
        for i in range(self.nchs):
            self.chnums.append(int(chs[i].name.split('_')[1].replace('ai','')))
            
    def get_raw(self):
        if self._param_name == 'volt_time':
            data = self.root_instrument.q.get()
            self.root_instrument.cnt_cont_mode += 1
            # return tuple(data[ch_num] for ch_num in self.chnums)
            return tuple(np.reshape(data[ch_num], (self.root_instrument._sample_interval(),1,)) for ch_num in self.chnums)
        elif self._param_name == 'volt': 
            self.root_instrument.reader.read_many_sample(self.root_instrument.read_buffer, 
                                                         self.root_instrument.num_of_samps_to_read_per_chan)
            samples_avg = np.mean(self.root_instrument.read_buffer, axis=1)
            return tuple(samples_avg[ch_num] for ch_num in self.chnums)
            
         
class DAQVoltageChannel(InstrumentChannel):
    def __init__(self,
                 parent: Instrument,
                 name: str,
                 ch_num: int) -> None:
                
        super().__init__(parent, name)        
        self.instrument = self.root_instrument
        self.ch_num = ch_num
        
        self.add_parameter(name = 'volt_time',
                           label = f'AI{self.ch_num}',
                           unit = 'V',
                           get_cmd = partial(self.root_instrument._get_volt_time, self.ch_num),
                           setpoints = (self.root_instrument.time_axis,), 
                           parameter_class = ParameterWithSetpoints,                        
                           vals = Arrays(shape=(self.root_instrument._sample_interval(),1,))
                           ) 
        
        self.add_parameter(name = 'volt',
                           label = f'AI{self.ch_num}',
                           unit = 'V',
                           get_cmd = partial(self.root_instrument._get_volt, self.ch_num),
                           )
                                   
    
class DAQAnalogInputs(Instrument):
    """
    QCoDeS driver for the NI USB-6363 DAQ.
    
    Args:
        name: instrument name
        dev_name: device name (e.g. 'Dev1') as found in NI-MAX       
        rate: sampling rate per channel in Hz
        chs: dictionary of analog input channels
        int_time: integration time in s
        min_val: min of input volt range 
            (-0.1, -0.2, -0.5, -1, -2, -5, or -10 [Default])
        max_val: max of input volt range 
            (0.1, 0.2, 0.5, 1, 2, 5, or 10 [Default])
        sample_mode_cont: generate samples continuously (true) or a finite 
            number of samples (false). Default: false
    """
    def __init__(self, 
                 name: str, 
                 dev_name: str, 
                 rate: Union[int, float],
                 chs: List[int],
                 int_time: Optional[float] = 0.1, 
                 min_val: Optional[float] = -10,
                 max_val: Optional[float] = +10, 
                 sample_mode_cont: Optional[bool] = False, 
                 **kwargs) -> None:

        super().__init__(name, **kwargs)
        
        self.dev_name = dev_name
        self.rate = rate
        self.chs = chs
        self.int_time = int_time
        self.sample_mode_cont = sample_mode_cont
        self.nchs = len(chs)
        self.min_val = min_val
        self.max_val = max_val
        
        self.metadata.update({'dev_name': self.dev_name,
                              'rate': f'{self.rate} Hz',
                              'channels': self.chs,
                              'int_time': self.int_time})
        
        self.start_task()
        
        self.add_parameter('time_axis',
                           unit = 's',
                           label = 'Time',
                           parameter_class = GeneratedSetPoints,
                           vals = Arrays(shape=(self._sample_interval(),1,)),
                           snapshot_exclude = True)
        
        channels = ChannelList(self, 
                               'Channels', 
                               DAQVoltageChannel,
                               snapshotable = False,
                               multichan_paramclass = DAQVoltageMultiChannel)
         
        for idx in self.chs:           
            channel = DAQVoltageChannel(self, 
                                        f'ai{idx}', 
                                        idx)
            channels.append(channel)
            self.add_submodule(f'ai{idx}', channel)
        channels.lock()
        self.add_submodule('channels', channels)
        
        self.connect_message()
       
    
    def set_timing(self, rate, int_time):
        self.rate = rate
        self.int_time = int_time
    
        if self.sample_mode_cont == False:
            self.num_of_samps_to_read_per_chan = int(np.round(rate*int_time))
            self.task.timing.cfg_samp_clk_timing(rate,
                                                 sample_mode = AcquisitionType.FINITE,
                                                 samps_per_chan = self.num_of_samps_to_read_per_chan)
        else: 
            self.num_of_samps_to_read_per_chan = self._sample_interval()
            self.task.timing.cfg_samp_clk_timing(rate,
                                                 sample_mode = AcquisitionType.CONTINUOUS)
            
            self.time_axis.vals = Arrays(shape=(self._sample_interval(),1,))
            self.time_axis.shape = (self._sample_interval(),1,)
            for ch in self.channels:
                ch.volt_time.vals = Arrays(shape=(self._sample_interval(),1,)) 
                ch.volt_time.shape = (self._sample_interval(),1,)
             
        self.task.timing.ai_conv_rate = 1.1*rate*self.nchs
        self.read_buffer = np.zeros((self.nchs, self.num_of_samps_to_read_per_chan), dtype = np.float64)
        
        self.metadata.update({'dev_name': self.dev_name,
                              'rate': f'{self.rate} Hz',
                              'channels': self.chs,
                              'int_time': self.int_time})
    
    
    def start_task(self):
        self.cnt_cont_mode = -1 # counter for continuous data acquisition
        self.q = queue.Queue() # constructor for a FIFO queue
        
        self.task = nidaqmx.Task()
           
        # create channels to measure volts
        for ch_num in self.chs:
            self.task.ai_channels.add_ai_voltage_chan(f'{self.dev_name}/ai{ch_num}', 
                                                      f'ai{ch_num}', 
                                                      min_val = self.min_val,
                                                      max_val = self.max_val)
        
        # set timing configurations
        self.task.timing.ai_conv_rate = 1.1*self.rate*self.nchs  
        # increase interchannel delay to avoid ghosting between DAQ channels 
        # when reading out multiple channels
        
        if self.sample_mode_cont == True:
            self.num_of_samps_to_read_per_chan = self._sample_interval()
            self.time_axis.vals = Arrays(shape=(self._sample_interval(),1,))
            self.time_axis.shape = (self._sample_interval(),1,)
            for ch in self.channels:
                ch.volt_time.vals = Arrays(shape=(self._sample_interval(),1,)) 
                ch.volt_time.shape = (self._sample_interval(),1,)
            self.task.timing.cfg_samp_clk_timing(self.rate,
                                                 sample_mode = AcquisitionType.CONTINUOUS)
            self.task.register_every_n_samples_acquired_into_buffer_event(self.num_of_samps_to_read_per_chan, 
                                                                          self._get_volt_cont)
            self.task.start()
        else:
            self.num_of_samps_to_read_per_chan = int(np.around(self.rate*self.int_time))
            self.task.timing.cfg_samp_clk_timing(self.rate,
                                                 sample_mode = AcquisitionType.FINITE,
                                                 samps_per_chan = self.num_of_samps_to_read_per_chan)

        # self.task.in_stream.auto_start=False
        self.reader = stream_readers.AnalogMultiChannelReader(self.task.in_stream)
        self.read_buffer = np.zeros((self.nchs, self.num_of_samps_to_read_per_chan), dtype = np.float64)
    
    
    def switch_sample_mode(self):
        # switch from FINITE to CONTINUOUS sampling or vice versa
        if self.sample_mode_cont == False:
            self.sample_mode_cont = True
            self.task.stop()
            self.task.close()
            self.start_task() 
        else:
            self.sample_mode_cont = False
            self.task.stop()
            self.task.close()
            self.start_task()
    
    
    def _get_volt(self, channel):
        # read samples
        self.reader.read_many_sample(self.read_buffer, 
                                     self.num_of_samps_to_read_per_chan)
        # average samples
        samples_avg = np.mean(self.read_buffer, axis=1)
        return samples_avg[channel]
    
    
    def _get_volt_time(self, channel):
        # start = time.perf_counter()
        data = self.q.get()
        # elapsed = time.perf_counter()-start
        # print(f"Time per reading: {elapsed:.3f} s")
        self.cnt_cont_mode += 1
        return np.reshape(data[channel],(self.num_of_samps_to_read_per_chan,1,))
    
    
    def _get_volt_cont(self, task_handle, event_type, number_of_samples, callback_data):
        ''' callback function that is executed when number_of_samples samples
            are in the buffer
        
            The function you pass must have the following prototype:

                >>> def callback(task_handle, every_n_samples_event_type,
                >>>         number_of_samples, callback_data):
                >>>     return 0
        '''      
        # read samples
        self.reader.read_many_sample(self.read_buffer, 
                                     self.num_of_samps_to_read_per_chan)
        ''' 
        TODO: despite setting self.task.in_stream.auto_start to False, 
              the task is automatically stopped after read
        
        if the task would still be running uncommenting the following line
        should raise an error
        '''
        # self.task.timing.ai_conv_rate = 1.1*self.rate*self.nchs
    
        # add samples to FIFO
        self.q.put(self.read_buffer)

        return 0
    
    
    def get_idn(self) -> Dict[str, Optional[str]]:
        
        '''
        Returns a dictionary with the device information
        '''
        system = nidaqmx.system.System.local()
        a = system.driver_version
        firmware = str(a[0])+'.'+str(a[1])+'.'+str(a[2])
        model = system.devices[0].product_type
        serial = system.devices[0].dev_serial_num
        return {"vendor": "NI", 
                "model": model,
                "serial": serial, 
                "firmware": firmware}
    
    
    def _sample_interval(self,
                         update_interval: Optional[float] = 0.1):
        '''
        Specifies the number of samples after which an event occurs 
        (continuous samples)
        
        NI-DAQmx will allocate a buffer size that depends on the sample rate:
            0-100 S/s -> 1 kS
            100-10,000 S/s -> 10 kS
            10,000-1,000,000 S/s -> 100 kS
            > 1,000,000 S/s -> 1 MS
        
         use self.task.in_stream.input_buf_size to get/set the number of 
         samples the input buffer can hold for each channel
         
         use self.task.in_stream.input_buf_size to query the size of the 
         onboard input buffer (FIFO) in samples per channel. The NI DAQ 6363 
         has a FIFO of 2047 samples
         
         every "update_interval" seconds the signal is read out
         
         For the continuous mode the buffer size needs to be an even multiple 
         of the "Every N Samples Interval", thus we have to adjust the buffer
         size
        
        '''
        self.update_interval = update_interval
        every_n_samples_interval = int(np.around(update_interval*self.rate))
        self.task.in_stream.input_buf_size = 10*every_n_samples_interval
        
        return every_n_samples_interval