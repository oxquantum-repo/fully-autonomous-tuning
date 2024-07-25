# Author: Mohammad Samani
# Date:   24.11.2021
# Place:  Basel, Switzerland
import numpy as np
from typing import Any, Optional
from qcodes import Instrument, VisaInstrument, Parameter
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A, AWGChannel, SRValidator, _parse_string_response
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes import validators as vals
from functools import partial
import broadbean as bb
from SPIN_qcodes_tools.Experiments.QubitSpectroscopy.QubitParameters import QubitParameters
import datetime, time

##################################################
#
# MODEL DEPENDENT SETTINGS
#
# TODO: it seems that a lot of settings differ between models
# perhaps these dicts should be merged to one

_fg_path_val_map = {'5204': {'DC High BW': "DCHB",
                             'DC High Voltage': "DCHV",
                             'AC Direct': "ACD"},
                    '5208': {'DC High BW': "DCHB",
                             'DC High Voltage': "DCHV",
                             'AC Direct': "ACD"},
                    '70001A': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'},
                    '70002A': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'},
                    '70001B': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'},
                    '70002B': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'}}

# number of markers per channel
_num_of_markers_map = {'5204': 4,
                       '5208': 4}

# channel resolution
_chan_resolutions = {'5204': [12, 13, 14, 15, 16],
                     '5208': [12, 13, 14, 15, 16]}

# channel resolution docstrings
_chan_resolution_docstrings = {'5204': "12 bit resolution allows for four "
                                       "markers, 13 bit resolution "
                                       "allows for three, etc. with 16 bit "
                                       "allowing for ZERO markers",
                               '5208': "12 bit resolution allows for four "
                                       "markers, 13 bit resolution "
                                       "allows for three, etc. with 16 bit "
                                       "allowing for ZERO markers"}
# channel amplitudes
_chan_amps = {'5204': 1.5,
              '5208': 1.5}

# marker ranges
_marker_high = {'5204': (-0.5, 1.75),
                '5208': (-0.5, 1.75)}
_marker_low = {'5204': (-0.3, 1.55),
               '5208': (-0.3, 1.55)}

class SRValidator5200(SRValidator):
    """
    Validator to validate the AWG clock sample rate
    """
    def __init__(self, awg: 'AWG5200') -> None:
        """
        Args:
            awg: The parent instrument instance. We need this since sample
                rate validation depends on many clock settings
        """
        self.awg = awg
        if self.awg.model in ['5204', '5208']:
            self._internal_validator = vals.Numbers(1.49e3, 2.5e9)

            
class AWG5200Channel(AWGChannel):
    """
    Class to hold a channel of the AWG.
    """

    def __init__(self,  parent: Instrument, name: str, channel: int) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The name used in the DataSet
            channel: The channel number, either 1 or 2.
        """

        InstrumentChannel.__init__(self, parent, name)

        self.channel = channel

        num_channels = self.root_instrument.num_channels
        self.model = self.root_instrument.model

        fg = 'function generator'

        if channel not in list(range(1, num_channels+1)):
            raise ValueError('Illegal channel value.')

        self.add_parameter('state',
                           label=f'Channel {channel} state',
                           get_cmd=f'OUTPut{channel}:STATe?',
                           set_cmd=f'OUTPut{channel}:STATe {{}}',
                           vals=vals.Ints(0, 1),
                           get_parser=int)

        ##################################################
        # FGEN PARAMETERS

        # TODO: Setting high and low will change this parameter's value
        self.add_parameter('fgen_amplitude',
                           label=f'Channel {channel} {fg} amplitude',
                           get_cmd=f'FGEN:CHANnel{channel}:AMPLitude?',
                           set_cmd=f'FGEN:CHANnel{channel}:AMPLitude {{}}',
                           unit='V',
                           vals=vals.Numbers(0, _chan_amps[self.model]),
                           get_parser=float)

        self.add_parameter('fgen_offset',
                           label=f'Channel {channel} {fg} offset',
                           get_cmd=f'FGEN:CHANnel{channel}:OFFSet?',
                           set_cmd=f'FGEN:CHANnel{channel}:OFFSet {{}}',
                           unit='V',
                           vals=vals.Numbers(0, 0.250),  # depends on ampl.
                           get_parser=float)

        self.add_parameter('fgen_frequency',
                           label=f'Channel {channel} {fg} frequency',
                           get_cmd=f'FGEN:CHANnel{channel}:FREQuency?',
                           set_cmd=partial(self._set_fgfreq, channel),
                           unit='Hz',
                           get_parser=float)

        self.add_parameter('fgen_dclevel',
                           label=f'Channel {channel} {fg} DC level',
                           get_cmd=f'FGEN:CHANnel{channel}:DCLevel?',
                           set_cmd=f'FGEN:CHANnel{channel}:DCLevel {{}}',
                           unit='V',
                           vals=vals.Numbers(-0.25, 0.25),
                           get_parser=float)

        self.add_parameter('fgen_signalpath',
                           label=f'Channel {channel} {fg} signal path',
                           set_cmd=f'FGEN:CHANnel{channel}:PATH {{}}',
                           get_cmd=f'FGEN:CHANnel{channel}:PATH?',
                           val_mapping=_fg_path_val_map[self.root_instrument.model])

        self.add_parameter('fgen_period',
                           label=f'Channel {channel} {fg} period',
                           get_cmd=f'FGEN:CHANnel{channel}:PERiod?',
                           unit='s',
                           get_parser=float)

        self.add_parameter('fgen_phase',
                           label=f'Channel {channel} {fg} phase',
                           get_cmd=f'FGEN:CHANnel{channel}:PHASe?',
                           set_cmd=f'FGEN:CHANnel{channel}:PHASe {{}}',
                           unit='degrees',
                           vals=vals.Numbers(-180, 180),
                           get_parser=float)

        self.add_parameter('fgen_symmetry',
                           label=f'Channel {channel} {fg} symmetry',
                           set_cmd=f'FGEN:CHANnel{channel}:SYMMetry {{}}',
                           get_cmd=f'FGEN:CHANnel{channel}:SYMMetry?',
                           unit='%',
                           vals=vals.Numbers(0, 100),
                           get_parser=float)

        self.add_parameter('fgen_type',
                           label=f'Channel {channel} {fg} type',
                           set_cmd=f'FGEN:CHANnel{channel}:TYPE {{}}',
                           get_cmd=f'FGEN:CHANnel{channel}:TYPE?',
                           val_mapping={'SINE': 'SINE',
                                        'SQUARE': 'SQU',
                                        'TRIANGLE': 'TRI',
                                        'NOISE': 'NOIS',
                                        'DC': 'DC',
                                        'GAUSSIAN': 'GAUSS',
                                        'EXPONENTIALRISE': 'EXPR',
                                        'EXPONENTIALDECAY': 'EXPD',
                                        'NONE': 'NONE'})

        ##################################################
        # AWG PARAMETERS

        # this command internally uses power in dBm
        # the manual claims that this command only works in AC mode
        # (OUTPut[n]:PATH is AC), but I've tested that it does what
        # one would expect in DIR mode.
        self.add_parameter(
            'awg_amplitude',
            label=f'Channel {channel} AWG peak-to-peak amplitude',
            set_cmd=f'SOURCe{channel}:VOLTage {{}}',
            get_cmd=f'SOURce{channel}:VOLTage?',
            unit='V',
            get_parser=float,
            vals=vals.Numbers(0.250, _chan_amps[self.model]))

        self.add_parameter('assigned_asset',
                           label=('Waveform/sequence assigned to '
                                  f' channel {self.channel}'),
                           get_cmd=f"SOURCE{self.channel}:CASSet?",
                           get_parser=_parse_string_response)
        
        # Added by M.S.
        self.add_parameter('current_step',
                           label=(f'Current step for channel {self.channel}'),
                           set_cmd=f"SOURce{self.channel}:JUMP:FORCe {{}}"
                           #get_cmd=f"SOURce{self.channel}:SCSTep?",
                           #get_parser=float
                          )

        # markers
        for mrk in range(1, _num_of_markers_map[self.model]+1):

            self.add_parameter(
                f'marker{mrk}_high',
                label=f'Channel {channel} marker {mrk} high level',
                set_cmd=partial(self._set_marker, channel, mrk, True),
                get_cmd=f'SOURce{channel}:MARKer{mrk}:VOLTage:HIGH?',
                unit='V',
                vals=vals.Numbers(*_marker_high[self.model]),
                get_parser=float)

            self.add_parameter(
                f'marker{mrk}_low',
                label=f'Channel {channel} marker {mrk} low level',
                set_cmd=partial(self._set_marker, channel, mrk, False),
                get_cmd=f'SOURce{channel}:MARKer{mrk}:VOLTage:LOW?',
                unit='V',
                vals=vals.Numbers(*_marker_low[self.model]),
                get_parser=float)

            self.add_parameter(
                f'marker{mrk}_waitvalue',
                label=f'Channel {channel} marker {mrk} wait state',
                set_cmd=f'OUTPut{channel}:WVALue:MARKer{mrk} {{}}',
                get_cmd=f'OUTPut{channel}:WVALue:MARKer{mrk}?',
                vals=vals.Enum('FIRST', 'LOW', 'HIGH'))

            self.add_parameter(
                name=f'marker{mrk}_stoppedvalue',
                label=f'Channel {channel} marker {mrk} stopped value',
                set_cmd=f'OUTPut{channel}:SVALue:MARKer{mrk} {{}}',
                get_cmd=f'OUTPut{channel}:SVALue:MARKer{mrk}?',
                vals=vals.Enum('OFF', 'LOW'))

        ##################################################
        # MISC.

        self.add_parameter('resolution',
                           label=f'Channel {channel} bit resolution',
                           get_cmd=f'SOURce{channel}:DAC:RESolution?',
                           set_cmd=f'SOURce{channel}:DAC:RESolution {{}}',
                           vals=vals.Enum(*_chan_resolutions[self.model]),
                           get_parser=int,
                           docstring=_chan_resolution_docstrings[self.model])
    
    def set_qubitparameter_value(self, value: float, dim: int):
        dic = self.root_instrument.doNd
        index = np.where(dic[dim]["range"] == value)[0][0]
        dic[dim]["current_index"] = index
        value_list = -1*np.ones(len(dic))
        for i in range(len(dic)):
            value_list[i] = dic[i]["range"][dic[i]["current_index"]]
        
        self.set_step_on_awg(value_list)
    
    def set_step_on_awg(self, value_list: list):
        dic = self.root_instrument.doNd
        shape = tuple([len(dic[i]["range"]) for i in range(len(dic))])
        if len(dic) < 1:
            raise Exception("You need to set up the doNd dictionary under the parent instrument first.")
        if len(value_list) != len(dic):
            raise Exception("The number of values must be the same as the dimensions of your scan.")
        indexes = -1*np.ones(len(dic))
        seq_index = 0
        indexes = np.array([np.where(dic[i]["range"] == value_list[i])[0][0] for i in range(len(dic))])
        flat_index = np.ravel_multi_index(indexes, shape)
        #print(f"flat_index={flat_index}, value list={value_list}")
        self.write(f"SOURce{self.channel}:JUMP:FORCe {flat_index+1}")
        
        

class AWG5200(AWG70000A):
    """
    The QCoDeS driver for Tektronix AWG5208, inherits from the driver provided by QCodes' main repository, AWG70000A
    """

    def __init__(self, name: str, address: str,
                 timeout: float=10, **kwargs: Any) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds)
        """
        VisaInstrument.__init__(self, name, address, timeout=timeout, terminator='\n', **kwargs)

        self.num_channels = int(self.ask("AWGControl:CONFigure:CNUMber?"))

        self.model = self.IDN()['model'][3:]
        
        self.doNdrange = []

        if self.model not in ['5208', '5204']:
            raise ValueError('Unknown model type: {}. Are you using '
                             'the right driver for your instrument?'
                             ''.format(self.model))

        self.add_parameter('current_directory',
                           label='Current file system directory',
                           set_cmd='MMEMory:CDIRectory "{}"',
                           get_cmd='MMEMory:CDIRectory?',
                           vals=vals.Strings())

        self.add_parameter('mode',
                           label='Instrument operation mode',
                           set_cmd='INSTrument:MODE {}',
                           get_cmd='INSTrument:MODE?',
                           vals=vals.Enum('AWG', 'FGEN'))

        ##################################################
        # Clock parameters        
        self.add_parameter('sample_rate',
                           label='Clock sample rate',
                           set_cmd='CLOCk:SRATe {}',
                           get_cmd='CLOCk:SRATe?',
                           unit='Sa/s',
                           get_parser=float,
                           vals=SRValidator5200(self))

        self.add_parameter('clock_source',
                           label='Clock source',
                           set_cmd='CLOCk:SOURce {}',
                           get_cmd='CLOCk:SOURce?',
                           val_mapping={'Internal': 'INT',
                                        'Internal, 10 MHZ ref.': 'EFIX',
                                        'Internal, variable ref.': 'EVAR',
                                        'External': 'EXT'})

        self.add_parameter('clock_external_frequency',
                           label='External clock frequency',
                           set_cmd='CLOCk:ECLock:FREQuency {}',
                           get_cmd='CLOCk:ECLock:FREQuency?',
                           get_parser=float,
                           unit='Hz',
                           vals=vals.Numbers(6.25e9, 12.5e9))

        self.add_parameter('run_state',
                           label='Run state',
                           get_cmd='AWGControl:RSTATe?',
                           val_mapping={'Stopped': '0',
                                        'Waiting for trigger': '1',
                                        'Running': '2'})
        
        # We deem 2 channels too few for a channel list
        if self.num_channels > 2:
            self.chanlist = ChannelList(self, 'Channels', AWG5200Channel,
                                   snapshotable=False)

        for ch_num in range(1, self.num_channels+1):
            ch_name = f'ch{ch_num}'
            channel = AWG5200Channel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            if self.num_channels > 2:
                self.chanlist.append(channel)

        if self.num_channels > 2:
            self.chanlist.lock()
            self.add_submodule('channels', self.chanlist)

        # Folder on the AWG where to files are uplaoded by default
        self.wfmxFileFolder = "\\Users\\OEM\\Documents"
        self.seqxFileFolder = "\\Users\\OEM\\Documents"

        self.current_directory(self.wfmxFileFolder)

        self.connect_message()
    
    def loadSEQXFile(self, filename: str, path: Optional[str] = None) -> None:
        """
        Load a seqx file from instrument disk memory. All sequences in the file
        are loaded into the sequence list.
        Args:
            filename: The name of the sequence file INCLUDING the extension
            path: Path to load from. If omitted, the default path
                (self.seqxFileFolder) is used.
        """
        if not path:
            path = self.seqxFileFolder

        pathstr = f'C:{path}\\{filename}'

        self.write(f'MMEMory:OPEN:SASSet:SEQuence "{pathstr}"')            
    
    def load_sequence(self, seq: bb.Sequence, qp: QubitParameters, filename: str='', play: bool=True) -> None:
        """
        Convert a broadbean sequence into a SEQX file and uplpad it onto the AWG, load it, and play it.
        I am not convinced this function belongs to the device driver's class of the AWG.
        """
        for i, ch in enumerate(qp.CIQ_chs):
            self.chanlist[i].awg_amplitude(qp.CIQ_champls[i])  # this is the peak-to-peak amplitude of the channel
            self.chanlist[i].resolution(14)    # We need two markers. 16-2=14        
        self.sample_rate(qp.sampling_rate)
        seqx_input = seq.outputForSEQXFile()
        seqx_output = self.makeSEQXFile(*seqx_input)
        
        if filename == '':
            filename = f"file_{datetime.datetime.now().strftime('%Y%m%d')}.seqx"
        
        self.sendSEQXFile(seqx_output, filename)
        self.loadSEQXFile(filename)      
        
        # assign tracks from the sequence to the awg sequencer
        for i, ch in enumerate(qp.CIQ_chs):
            self.chanlist[ch-1].setSequenceTrack(seq.name, i+1)
            self.chanlist[ch-1].state(1)
        
        if play:
            self.play()       
        
