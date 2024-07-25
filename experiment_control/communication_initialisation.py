# %matplotlib notebook
# %matplotlib inline
import os
import time
from datetime import datetime
from pprint import pprint
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

# qcodes imports
import qcodes as qc
from qcodes import (
    ManualParameter,
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_guid,
    load_by_run_spec,
    load_experiment,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
from qcodes.dataset.plotting import plot_dataset
from qcodes.instrument.parameter import (
    Parameter,
    ParameterWithSetpoints,
    ScaledParameter,
)
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter
from qcodes.interactive_widget import experiments_widget
from qcodes.logger.logger import start_all_logging
from qcodes.tests.instrument_mocks import (
    DummyInstrument,
    DummyInstrumentWithMeasurement,
)
from qcodes.utils.dataset.doNd import do0d, do1d, do2d
from qcodes.utils.metadata import diff_param_values
from qcodes.utils.validators import Arrays, Numbers

#for AWG waves
import broadbean as bb
import nidaqmx

# Import instrument drivers
#from qcodes.instrument_drivers.agilent.Agilent_34400A import Agilent_34400A # Agilent DMM 34411A
#from qcodes.instrument_drivers.Keysight.Keysight_34465A_submodules import Keysight_34465A # Keysight DMM 34465A
import zhinst.qcodes as ziqc  # ZI MFLI lock-in amplifier
from broadbean.plotting import plotter
from qcodes.instrument_drivers.tektronix.Keithley_2400 import Keithley_2400
from qcodes_addons.AWGhelp import PulseParameter

#from ANC350 import ANC350 # attocube controller
#from ANC350Lib.v4 import ANC350v4Lib
#libfile = ANC350v4Lib(mydir+r"\anc350v4.dll")
from qcodes_addons.BaselAWG5204 import BaselAWG5204  # Modified AWG

# Import helper functions
from qcodes_addons.Datahelp import (
    dataset_to_numpy,
    datasetID_to_numpy,
    define_detuning,
    plot2D_by_ID,
)
from qcodes_addons.doNdAWG import do1dAWG, do2dAWG, init_Rabi
from qcodes_addons.E8257D import Agilent_E8527D
from qcodes_addons.Lakeshore_331 import Model_331
from qcodes_addons.NIDAQ import DAQAnalogInputs  # NI DAQ
from qcodes_addons.Oxford_Instruments_IPS120 import OxfordInstruments_IPS120
from qcodes_addons.Parameterhelp import (
    AxisParameter,
    CompensatedGateParameter,
    GateParameter,
    MultiDAQParameter,
    VirtualGateParameter,
    ZILockinParameter,
)
from qcodes_addons.SGS100A import RohdeSchwarz_SGS100A
from qcodes_addons.SP927v1 import SP927  # Basel DAC


def get_comms(db_name = "GeSiNW_Qubit_VTI01_Miguel.db", sample_name = "Butch", exp_name = "Qxford_AI_Optimization"):
    # Create or load database
    db_file_path = os.path.join(os.getcwd(), db_name)
    qc.config.core.db_location = db_file_path
    initialise_or_create_database_at(db_file_path)

    experiment = load_or_create_experiment(experiment_name=exp_name,
                                           sample_name=sample_name)

    station = qc.Station()

    # Defining Pulse Parameters
    pp = PulseParameter(t_RO=50e-9,  # readout part
                        t_CB=50e-9,  # coulomb blockade part
                        t_ramp=4e-9,
                        t_burst=4e-9,
                        C_ampl=0,
                        I_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
                        Q_ampl=0.3,  # 0 to 1 or 0.5 (?) normalised, going to the vector source and scaling its output
                        IQ_delay=19e-9,  #
                        f_SB=0,  # sideband modulation, can get you better signal or enable 2 qubit gates
                        f_lockin=87.77,  # avoid 50Hz noise by avoiding multiples of it
                        CP_correction_factor=0.848)  # triangle splitting over C_ampl, how much of the pulse arrives at the sample [in mV/mV]
    # old compensation from MJC before was 0.386 mV/mV (IGOR NEEDS DIV BY 2, here take total splitting !!),
    # we rechecked based on scan #49 to improve compensation 230428

    # BasPI DAC 8-chanel
    dac = SP927('LNHR_dac', 'ASRL2::INSTR')
    station.add_component(dac)

    # MFLI Zurich Instruments
    mfli = ziqc.MFLI("dev5240", host="127.0.0.1", name="mf1i", interface="USB", )
    station.add_component(mfli)

    daq_chs = [0, 1, 2, 3, 4]
    # Specify the DAQ channels to be recorded
    # Channel numbers as stated on the DAQ front panel
    DAQ = DAQAnalogInputs(name='daq',
                          dev_name='Dev2',  # Device name can be found in NI-MAX
                          rate=1000,
                          chs=daq_chs,
                          int_time=0.125)
    station.add_component(DAQ)

    # IPS120 Oxford Instruments (using RS232)
    ips = OxfordInstruments_IPS120(name="IPS", address="GPIB0::10::INSTR", use_gpib=True, number=2, baud_rate=9600,
                                   data_bits=8, )
    station.add_component(ips)
    ips.initialise_magnet()

    # 4-channel Tektronics AWG5204
    awg = BaselAWG5204('awg',
                       'TCPIP0::192.168.10.2::INSTR')  # driver adapted from 5208 version, setting configs in AWG70000A the same as for 5204 as for 5208
    # awg = BaselAWG5204('awg', 'USB0::0x0699::0x0503::B030570::INSTR') #driver adapted from 5208 version, setting configs in AWG70000A the same as for 5204 as for 5208
    station.add_component(awg)

    # RS SGS100A Vector Source
    VS = RohdeSchwarz_SGS100A("VS", "USB0::0x0AAD::0x0088::110184::INSTR")
    station.add_component(VS)

    # Initializing DC parameters like gate voltages & DC current readout: Low T

    VSD = GateParameter(dac.ch8.volt,
                        name="V_SD",
                        unit="V",
                        value_range=(-6, 6),
                        scaling=103.8,
                        offset=0)

    # VD = GateParameter(dac.ch7.volt,
    #                    name = "V_D",
    #                    unit = "V",
    #                    value_range = (-6, 6),
    #                    scaling = 103,
    #                    offset=0)

    # VSD = VirtualGateParameter(name = "V_SD",
    #                    params = (VS, VD),
    #                    set_scaling = (1, 0),
    #                    get_scaling = 1)

    VL = GateParameter(dac.ch1.volt,
                       name="V_L",
                       unit="V",
                       value_range=(-1, 2))

    VLP = GateParameter(dac.ch2.volt,
                        name="V_LP",
                        unit="V",
                        value_range=(-1, 2))

    VM = GateParameter(dac.ch3.volt,
                       name="V_M",
                       unit="V",
                       value_range=(-1, 2))

    # VRP = GateParameter(dac.ch4.volt,
    #                     name = "V_RP",
    #                     unit = "V",
    #                     value_range = (-1,2))

    # Gates with compensation for RF pulses

    VRP = CompensatedGateParameter(dac.ch4.volt,
                                   name="V_RP",
                                   value_range=(-1, 2),
                                   pp=pp)

    VR = GateParameter(dac.ch5.volt,
                       name="V_R",
                       unit="V",
                       value_range=(-1, 2))

    # all plunger gates
    VPall = VirtualGateParameter(name="V_Pall",
                                 params=(VLP, VRP),
                                 set_scaling=(1, 1),
                                 offsets=(0, 0))
    # all nano-gates
    Vall = VirtualGateParameter(name="V_all",
                                params=(VLP, VRP, VL, VM, VR),
                                set_scaling=(1, 1, 1, 1, 1),
                                offsets=(0, 0, 0, 0, 0))

    # Specify IV converter gains
    gain_SD = ManualParameter('gain_SD',
                              initial_value=1 * 1e9,
                              unit='V/A')

    # gain_gate = ManualParameter('gain_gate',
    #                            initial_value = 1*1e6,
    #                            unit = 'V/A')

    # IS = ScaledParameter(DAQ.ai0.volt,
    #                      name = "I_S",
    #                      division = gain_SD,
    #                      unit = "A")

    # ID = ScaledParameter(DAQ.ai1.volt,
    #                      name = "I_D",
    #                      division = gain_SD,
    #                      unit = "A")

    # ISD = MultiDAQParameter([IS, ID], "I_SD")

    # IPLUS = ScaledParameter(DAQ.ai0.volt,
    #                      name = "I_NOISE",
    #                      division = gain_SD,
    #                      unit = "A")

    ISD = ScaledParameter(DAQ.ai0.volt,
                          name="I_SD",
                          division=gain_SD,
                          unit="A")

    # Iall = MultiDAQParameter([IPLUS, ISD, IB, IP1, IP2, IL, ISB1, ISB2], "I_all")

    # initializing MFLI Lock-In parameters

    LIfreq = GateParameter(mfli.oscs[0].freq,
                           name="LI_freq",
                           unit="Hz",
                           value_range=(0, 600e6),
                           scaling=1,
                           offset=0)

    LITC = GateParameter(mfli.demods[0].timeconstant,
                         name="LI_TC",
                         unit="s",
                         value_range=(0, 10),
                         scaling=1,
                         offset=0)

    LIXY = ZILockinParameter(mfli, ['X', 'Y'],
                             'XY',
                             names=['LI_X', 'LI_Y'],
                             gain=gain_SD,
                             scaling=1,
                             units=['A', 'A'])

    LIXYRPhi = ZILockinParameter(mfli,
                                 ['X', 'Y', 'R', 'Phase'],
                                 'XYRPhi',
                                 names=['LIX', 'LIY', 'LIR', 'LIPhase'],
                                 gain=1,
                                 scaling=1,
                                 units=['A', 'A', 'A', '°'])

    LIX = ZILockinParameter(mfli,
                            ['X'],
                            'X',
                            names=['LIX'],
                            gain=gain_SD,
                            scaling=1,
                            units=['A'])

    LIY = ZILockinParameter(mfli,
                            ['Y'],
                            'Y',
                            names=['LIY'],
                            gain=gain_SD,
                            scaling=1,
                            units=['A'])

    LIR = ZILockinParameter(mfli,
                            ['R'],
                            'R',
                            names=['LIR'],
                            gain=gain_SD,
                            scaling=1,
                            units=['A'])

    LIPhi = ZILockinParameter(mfli,
                              ['Phase'],
                              'Phi',
                              names=['LIPhase'],
                              gain=1,
                              scaling=1,
                              units=['°'])

    LIPhaseAdjust = GateParameter(mfli.demods[0].phaseadjust,
                                  name="Phase_adjust",
                                  unit="s",
                                  value_range=(0, 1),
                                  scaling=1,
                                  offset=0)

    # this is the "auto-phase adjust" command
    LIPhaseAdjust(1)
    # manually set phase of lockin
    # mfli.demods[0].phaseshift(-17)

    # next: configure for external reference operation
    # mfli.extrefs[0].enable(0)
    mfli.oscs[0].freq(87.777)
    mfli.triggers.out[0].source(1)
    mfli.demods[1].harmonic(1)
    # mfli.demods[1].harmonic(2)
    mfli.demods[1].adcselect(0)
    # other config
    mfli.sigins[0].range(300e-3)
    mfli.sigins[0].scaling(1)
    # mfli.demods[0].order(1)
    LITC(0.5)
    # mfli.demods[0].sinc(1)

    # RS Vector soucre parameters
    VS_freq = GateParameter(VS.frequency,
                            name="VS_freq",
                            unit="HZ",
                            value_range=(1e6, 20e9))

    VS_phase = GateParameter(VS.phase,
                             name="VS_phase",
                             unit="deg",
                             value_range=(0, 360))

    VS_pwr = GateParameter(VS.power,
                           name="VS_power",
                           unit="dbm",
                           value_range=(-120, 25))

    def VS_pulse_on():
        VS.pulsemod_state(1)

    def VS_pulse_off():
        VS.pulsemod_state(0)

    def VS_IQ_on():
        VS.IQ_state(1)

    def VS_IQ_off():
        VS.IQ_state(0)

    def VS_status():
        print("Output frequency:", VS_freq() / 1e6, "MHz", "\n"
                                                           "Output_power:", VS_pwr(), "dBm", "\n"
                                                                                             "Output state:",
              VS.status(), "\n"
                           "Pulse Modulation:", VS.pulsemod_state(), "\n"
                                                                     "IQ Modulation:", VS.IQ_state(), "\n")

    def detuning(x1, y1, x2, y2):
        slope = (y1 - y2) / (x1 - x2)
        intercept = (y1 + y2 - slope * (x1 + x2)) / 2
        return [slope, intercept]

    return [station, dac, DAQ, mfli, ips, awg, VS], [pp, VLP, VRP, VL, VM, VR, Vall, ISD, LITC, LIfreq, LIXY, LIXYRPhi, LIX, LIY, LIR, LIPhi, LIPhaseAdjust, VS_freq, VS_phase, VS_pwr, VS_pulse_on, VS_pulse_off, VS_IQ_on, VS_IQ_off, VS_status, detuning]