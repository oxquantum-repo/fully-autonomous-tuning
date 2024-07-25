from typing import Optional, Sequence, Dict, Tuple, Any, Union, List
import time
import pyvisa as visa
import logging
from functools import partial
from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.instrument.channel import MultiChannelInstrumentParameter
from qcodes.utils import validators as vals
log = logging.getLogger(__name__)

class SP927Exception(Exception):
    pass


class SP927Reader(object):
    def _vval_to_dacval(self, vval):
        """
        Convert voltage to DAC value 
        dacval=(Vout+10)*838848
        """
        dacval = int((float(vval)+10)*838848)
        return dacval

    def _dacval_to_vval(self, dacval):
        """
        Convert DAC value to voltage
        Vout=(dacval/838848)–10
        """
        vval = round((int(dacval.strip(),16)/float(838848))-10, 6)
        return vval


class SP927MultiChannel(MultiChannelInstrumentParameter, SP927Reader):
    def __init__(self, channels:Sequence[InstrumentChannel], param_name: str, *args: Any, **kwargs: Any):
        super().__init__(channels, param_name, *args, **kwargs)
        self._channels = channels
        self._param_name = param_name
        
        def get_raw(self):
            output = tuple(chan.parameters[self._param_name].get() for chan in self._channels)
            return output
        
        def set_raw(self, value):
            for chan in self._channels:
                chan.volt.set(value)
            
    
class SP927Channel(InstrumentChannel, SP927Reader):
   
    def __init__(self, parent, name, channel, min_val=-10, max_val=10):
        super().__init__(parent, name)
        
        # validate channel number
        self._CHANNEL_VAL = vals.Ints(1,8)
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel

        # limit voltage range
        self._volt_val = vals.Numbers(min(min_val, max_val), max(min_val, max_val))
        
        self.add_parameter('volt',
                           label = 'C {}'.format(channel),
                           unit = 'V',
                           set_cmd = partial(self._parent._set_voltage, channel),
                           set_parser = self._vval_to_dacval,
                           get_cmd = partial(self._parent._read_voltage, channel),
                           vals = self._volt_val 
                           )

class SP927(VisaInstrument, SP927Reader):
    """
    QCoDeS driver for the Basel Precision Instruments SP927 LNHR DAC
    https://www.baspi.ch/low-noise-high-resolution-dac
    """
    
    def __init__(self, name, address, min_val=-10, max_val=10, baud_rate=115200, **kwargs):
        """
        Creates an instance of the SP927 LNHR DAC instrument.

        Args:
            name (str): What this instrument is called locally.

            port (str): The address of the DAC. For a serial port this is ASRLn::INSTR
                        where n is replaced with the address set in the VISA control panel.
                        Baud rate and other serial parameters must also be set in the VISA control
                        panel.

            min_val (number): The minimum value in volts that can be output by the DAC.
            max_val (number): The maximum value in volts that can be output by the DAC.
        """
        super().__init__(name, address, **kwargs)

        # Serial port properties
        handle = self.visa_handle
        handle.baud_rate = baud_rate
        handle.parity = visa.constants.Parity.none
        handle.stop_bits = visa.constants.StopBits.one
        handle.data_bits = 8
        handle.flow_control = visa.constants.VI_ASRL_FLOW_XON_XOFF
        handle.write_termination = '\n'
        handle.read_termination = '\r\n'

        # Create channels
        channels = ChannelList(self, 
                               "Channels", 
                               SP927Channel, 
                               snapshotable = False,
                               multichan_paramclass = SP927MultiChannel)
        self.num_chans = 8
        
        for i in range(1, 1+self.num_chans):
            channel = SP927Channel(self, 'chan{:1}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{:1}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)

        # Safety limits for sweeping DAC voltages
        # inter_delay: Minimum time (in seconds) between successive sets.
        #              If the previous set was less than this, it will wait until the
        #              condition is met. Can be set to 0 to go maximum speed with
        #              no errors.    
         
        # step: max increment of parameter value.
        #       Larger changes are broken into multiple steps this size.
        #       When combined with delays, this acts as a ramp.
        for chan in self.channels:
            chan.volt.inter_delay = 0.2
            chan.volt.step = 0.050
        
        # switch all channels ON if still OFF
        if 'OFF' in self.query_all():
            self.all_on()
            
        self.connect_message()
        print('Current DAC output: ' +  str(self.channels[:].volt.get()))

    def _set_voltage(self, chan, code):
        self.write('{:0} {:X}'.format(chan, code))
            
    def _read_voltage(self, chan):
        dac_code=self.write('{:0} V?'.format(chan))
        return self._dacval_to_vval(dac_code)

    def set_all(self, volt):
        """
        Set all dac channels to a specific voltage.
        """
        for chan in self.channels:
            chan.volt.set(volt)
    
    def query_all(self):
        """
        Query status of all DAC channels
        """
        reply = self.write('All S?')
        return reply.replace("\r\n","").split(';')
    
    def all_on(self):
        """
        Turn on all channels.
        """
        self.write('ALL ON')
      
    def all_off(self):
        """
        Turn off all channels.
        """
        self.write('ALL OFF')
    
    def empty_buffer(self):
        # make sure every reply was read from the DAC 
        while self.visa_handle.bytes_in_buffer:
            print(self.visa_handle.bytes_in_buffer)
            print("Unread bytes in the buffer of DAC SP927 have been found. Reading the buffer ...")
            print(self.visa_handle.read_raw())
            # self.visa_handle.read_raw()
            print("... done")
            
    def write(self, cmd):
        """
        Since there is always a return code from the instrument, we use ask instead of write
        TODO: interpret the return code (0: no error)
        """
        # make sure there is nothing in the buffer
        self.empty_buffer()  
        
        return self.ask(cmd)
    
    def get_serial(self):
        """
        Returns the serial number of the device
        Note that when querying "HARD?" multiple statements, each terminated
        by \r\n are returned, i.e. the device`s reply is not terminated with 
        the first \n received

        """
        self.write('HARD?')
        reply = self.visa_handle.read()
        time.sleep(0.01)
        while self.visa_handle.bytes_in_buffer:
            self.visa_handle.read_raw()
            time.sleep(0.01)
        return reply.strip()[3:]
    
    def get_firmware(self):
        """
        Returns the firmware of the device
        Note that when querying "HARD?" multiple statements, each terminated
        by \r\n are returned, i.e. the device`s reply is not terminated with 
        the first \n received

        """
        self.write('SOFT?')
        reply = self.visa_handle.read()
        time.sleep(0.01)
        while self.visa_handle.bytes_in_buffer:
            self.visa_handle.read_raw()
            time.sleep(0.01)
        return reply.strip()[-5:]
        
    
    def get_idn(self):
        SN = self.get_serial()
        FW = self.get_firmware()
        return dict(zip(('vendor', 'model', 'serial', 'firmware'), 
                        ('BasPI', 'LNHR DAC SP927', SN, FW)))


if __name__ == '__main__':    
    dac = SP927('LNHR_dac3', 'ASRL19::INSTR')
    dac.ch1.volt.set(0.012)
    print(dac.ch1.volt.get())
    dac.close()
