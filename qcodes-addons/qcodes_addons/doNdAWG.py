# Author: Simon Geyer based on Moe's code
# Date:   12/01/2023
# Place:  Basel, Switzerland
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from qcodes import Parameter, config
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.measurements import Measurement
from qcodes.utils.dataset.doNd import (
    ActionsT,
    AxesTupleListWithDataSet,
    ParamMeasT,
    do1d,
    do2d,
)
from qcodes_addons.AWGhelp import (
    GenerateHahnSequence,
    GenerateRabiSequence,
    GenerateRamseySequence,
    PulseParameter,
    SequencePlotter,
)
from qcodes_addons.BaselAWG5204 import BaselAWG5204
from qcodes_addons.Parameterhelp import CompensatedGateParameter


def do1dAWG(pulse_type: str, vary: str, start: float, stop: float, num_points: int, delay: float, *param_meas: ParamMeasT, pp: PulseParameter, awg: BaselAWG5204, cgp: CompensatedGateParameter,
    enter_actions: ActionsT = (),
    exit_actions: ActionsT = (),
    write_period: Optional[float] = None,
    measurement_name: str = "",
    exp: Optional[Experiment] = None,
    do_plot: Optional[bool] = None,
    use_threads: Optional[bool] = None,
    additional_setpoints: Sequence[ParamMeasT] = tuple(),
    show_progress: Optional[None] = None,
    log_info: Optional[str] = None,
    show_pulse: Optional[bool] = True,
    skip_upload: Optional[bool] = False) -> AxesTupleListWithDataSet:
    """
    Parameters:
        pulse_type: type of qubit experiment
        vary: the name of the parameter in QubitParameters to scan
        start: the first point
        stop: the last point
        num_points
        delay
        param_meas
        pp: The PulseParameters that will be used as the default set. Only one aspect of it, indicated in "vary" will vary.
        awg: The AWG instrument. It can only be an AWG520x
    """
    rng = np.linspace(start, stop, num_points)
    
    #create Experiment waveform
    #Rabi experiments:
    if pulse_type == "Rabi":
        seqname = f"do1dAWG_Rabi_{vary}"
        seqx_input,original_length = GenerateRabiSequence(pp, vary, rng, seq_name=seqname)
    #Ramsey experiments:
    elif pulse_type == "Ramsey":
        seqname = f"do1dAWG_Ramsey_{vary}"
        seqx_input,original_length = GenerateRamseySequence(pp, vary, rng, seq_name=seqname)
    elif pulse_type == "Hahn":
        seqname = f"do1dAWG_Hahn_{vary}"
        seqx_input,original_length = GenerateHahnSequence(pp, vary, rng, seq_name=seqname)
    else:
        raise Exception("Unknown pulse type.")
    
    
    #plot sequence
    if show_pulse==True:
        SequencePlotter(seqx_input, pp, original_length)

    if not skip_upload:
        #create seqx file
        seqx = awg.makeSEQXFile(*seqx_input)
        filename = seqname+'.seqx'

        #Transfer the seqx file
        awg.sendSEQXFile(seqx, filename)
        print(f"Sequence file of size {len(seqx)/1024/1024:.3f} MB was sent to AWG")

        #Load the seqx file
        awg.loadSEQXFile(filename)

        #Assign tracks from the sequence to the channels
        awg.ch1.setSequenceTrack(seqname, 1)
        awg.ch2.setSequenceTrack(seqname, 2)
        awg.ch3.setSequenceTrack(seqname, 3)
    else:
        print('Skipping upload')
    #play!
    awg.ch1.state(1)
    awg.ch2.state(1)
    awg.ch3.state(1)
    awg.play()
    print("AWG is",awg.run_state())
    
    awg.wait_for_operation_to_complete()
    time.sleep(10)
    
    #keep track of the indices for each part of the sequence
    doNd = {val: 2*ind+1 for ind,val in enumerate(rng)}
    def set_channel(val):
        ind = doNd[val]
        awg.current_step_ch1(ind)
        awg.current_step_ch2(ind)
        awg.current_step_ch3(ind)
        if vary == "C_ampl":
            pp.C_ampl = val
            for gate in cgp:
                gate.update()
    
    setparam = Parameter(vary, label=PulseParameter.labels[vary],
                         unit=PulseParameter.units[vary],
                         set_cmd=set_channel)
        
    result = do1d(setparam, start, stop, num_points, delay, *param_meas,
                enter_actions=enter_actions,
                exit_actions=exit_actions,
                write_period=write_period,
                measurement_name=measurement_name,
                exp=exp,
                do_plot=do_plot,
                use_threads=use_threads,
                additional_setpoints=additional_setpoints,
                show_progress=show_progress,
                log_info=log_info
               )
    return result

def load_do1dAWG(pulse_type: str, vary: str, start: float, stop: float, num_points: int, delay: float,
            *param_meas: ParamMeasT, pp: PulseParameter, awg: BaselAWG5204, cgp: CompensatedGateParameter,
            enter_actions: ActionsT = (),
            exit_actions: ActionsT = (),
            write_period: Optional[float] = None,
            measurement_name: str = "",
            exp: Optional[Experiment] = None,
            do_plot: Optional[bool] = None,
            use_threads: Optional[bool] = None,
            additional_setpoints: Sequence[ParamMeasT] = tuple(),
            show_progress: Optional[None] = None,
            log_info: Optional[str] = None,
            show_pulse: Optional[bool] = True) -> AxesTupleListWithDataSet:
    """
        Parameters:
            pulse_type: type of qubit experiment
            vary: the name of the parameter in QubitParameters to scan
            start: the first point
            stop: the last point
            num_points
            delay
            param_meas
            pp: The PulseParameters that will be used as the default set. Only one aspect of it, indicated in "vary" will vary.
            awg: The AWG instrument. It can only be an AWG520x
        """
    rng = np.linspace(start, stop, num_points)

    # create Experiment waveform
    # Rabi experiments:
    if pulse_type == "Rabi":
        seqname = f"do1dAWG_Rabi_{vary}"
        seqx_input, original_length = GenerateRabiSequence(pp, vary, rng, seq_name=seqname)
    # Ramsey experiments:
    elif pulse_type == "Ramsey":
        seqname = f"do1dAWG_Ramsey_{vary}"
        seqx_input, original_length = GenerateRamseySequence(pp, vary, rng, seq_name=seqname)
    elif pulse_type == "Hahn":
        seqname = f"do1dAWG_Hahn_{vary}"
        seqx_input, original_length = GenerateHahnSequence(pp, vary, rng, seq_name=seqname)
    else:
        raise Exception("Unknown pulse type.")

    # plot sequence
    if show_pulse == True:
        SequencePlotter(seqx_input, pp, original_length)

    # create seqx file
    seqx = awg.makeSEQXFile(*seqx_input)
    filename = seqname + '.seqx'

    # Transfer the seqx file
    awg.sendSEQXFile(seqx, filename)
    print(f"Sequence file of size {len(seqx) / 1024 / 1024:.3f} MB was sent to AWG")

    # Load the seqx file
    awg.loadSEQXFile(filename)

    # Assign tracks from the sequence to the channels
    awg.ch1.setSequenceTrack(seqname, 1)
    awg.ch2.setSequenceTrack(seqname, 2)
    awg.ch3.setSequenceTrack(seqname, 3)

    # play!
    awg.ch1.state(1)
    awg.ch2.state(1)
    awg.ch3.state(1)
    awg.play()
    print("AWG is", awg.run_state())

    awg.wait_for_operation_to_complete()
    time.sleep(10)
    return

def run_do1dAWG(pulse_type: str, vary: str, start: float, stop: float, num_points: int, delay: float,
            *param_meas: ParamMeasT, pp: PulseParameter, awg: BaselAWG5204, cgp: CompensatedGateParameter,
            enter_actions: ActionsT = (),
            exit_actions: ActionsT = (),
            write_period: Optional[float] = None,
            measurement_name: str = "",
            exp: Optional[Experiment] = None,
            do_plot: Optional[bool] = None,
            use_threads: Optional[bool] = None,
            additional_setpoints: Sequence[ParamMeasT] = tuple(),
            show_progress: Optional[None] = None,
            log_info: Optional[str] = None,
            show_pulse: Optional[bool] = True) -> AxesTupleListWithDataSet:

    rng = np.linspace(start, stop, num_points)

    # create Experiment waveform
    # Rabi experiments:
    if pulse_type == "Rabi":
        seqname = f"do1dAWG_Rabi_{vary}"
        seqx_input, original_length = GenerateRabiSequence(pp, vary, rng, seq_name=seqname)
    # Ramsey experiments:
    elif pulse_type == "Ramsey":
        seqname = f"do1dAWG_Ramsey_{vary}"
        seqx_input, original_length = GenerateRamseySequence(pp, vary, rng, seq_name=seqname)
    elif pulse_type == "Hahn":
        seqname = f"do1dAWG_Hahn_{vary}"
        seqx_input, original_length = GenerateHahnSequence(pp, vary, rng, seq_name=seqname)
    else:
        raise Exception("Unknown pulse type.")

    # keep track of the indices for each part of the sequence
    doNd = {val: 2 * ind + 1 for ind, val in enumerate(rng)}

    def set_channel(val):
        ind = doNd[val]
        awg.current_step_ch1(ind)
        awg.current_step_ch2(ind)
        awg.current_step_ch3(ind)
        if vary == "C_ampl":
            pp.C_ampl = val
            for gate in cgp:
                gate.update()

    setparam = Parameter(vary, label=PulseParameter.labels[vary],
                         unit=PulseParameter.units[vary],
                         set_cmd=set_channel)

    result = do1d(setparam, start, stop, num_points, delay, *param_meas,
                  enter_actions=enter_actions,
                  exit_actions=exit_actions,
                  write_period=write_period,
                  measurement_name=measurement_name,
                  exp=exp,
                  do_plot=do_plot,
                  use_threads=use_threads,
                  additional_setpoints=additional_setpoints,
                  show_progress=show_progress,
                  log_info=log_info
                  )
    return result

def do2dAWG(pulse_type: str,
    vary1, start1: float, stop1: float, num_points1: int, delay1: float,
    vary2, start2: float, stop2: float, num_points2: int, delay2: float,
    *param_meas: ParamMeasT, pp: PulseParameter, awg: BaselAWG5204, cgp: CompensatedGateParameter,
    set_before_sweep: Optional[bool] = True,
    enter_actions: ActionsT = (), exit_actions: ActionsT = (),
    before_inner_actions: ActionsT = (), after_inner_actions: ActionsT = (), write_period: Optional[float] = None,
    measurement_name: str = "", exp: Optional[Experiment] = None, flush_columns: bool = False,
    do_plot: Optional[bool] = None, use_threads: Optional[bool] = None, additional_setpoints: Sequence[ParamMeasT] = tuple(),
    show_progress: Optional[None] = None, show_pulse: Optional[None] = None, log_info: Optional[str] = None) -> AxesTupleListWithDataSet:
    """
    Perform a 1D scan of ``param_set1`` from ``start1`` to ``stop1`` in
    ``num_points1`` and ``param_set2`` from ``start2`` to ``stop2`` in
    ``num_points2`` measuring param_meas at each step.
    Args:
        param_set1: The QCoDeS parameter to sweep over in the outer loop
        start1: Starting point of sweep in outer loop
        stop1: End point of sweep in the outer loop
        num_points1: Number of points to measure in the outer loop
        delay1: Delay after setting parameter in the outer loop
        param_set2: The QCoDeS parameter to sweep over in the inner loop
        start2: Starting point of sweep in inner loop
        stop2: End point of sweep in the inner loop
        num_points2: Number of points to measure in the inner loop
        delay2: Delay after setting parameter before measurement is performed
    Returns:
        The QCoDeS dataset.
    """
    is_AWG_parameter1 = isinstance(vary1, str)
    is_AWG_parameter2 = isinstance(vary2, str)
    
    if is_AWG_parameter1 and is_AWG_parameter2:
        #go into Rabi 2D
        print("Stop! this is not implemented yet!!!!")
        

        
    elif is_AWG_parameter1:
        rng1 = np.linspace(start1, stop1, num_points1)
        
        #keep track of the indices for each part of the sequence
        doNd = {val: 2*ind+1 for ind,val in enumerate(rng1)}
        def set_channel(val):
            ind = doNd[val]
            awg.current_step_ch1(ind)
            awg.current_step_ch2(ind)
            awg.current_step_ch3(ind)
            if vary1 == "C_ampl":
                pp.C_ampl = val
                for gate in cgp:
                    gate.update()
        
        setparam1 = Parameter(vary1, label=PulseParameter.labels[vary1],
                         unit=PulseParameter.units[vary1],
                         set_cmd=set_channel)    
        
        setparam2 = vary2
        
        #create Experiment waveform
        #Rabi experiments:
        if pulse_type == "Rabi":
            seqname = f"do2dAWG_Rabi_{vary1}"
            seqx_input,original_length = GenerateRabiSequence(pp, vary1, rng1, seq_name=seqname)
        #Ramsey experiments:
        elif pulse_type == "Ramsey":
            seqname = f"do2dAWG_Ramsey_{vary1}"
            seqx_input,original_length = GenerateRamseySequence(pp, vary1, rng1, seq_name=seqname)
        elif pulse_type == "Hahn":
            seqname = f"do2dAWG_Hahn_{vary1}"
            seqx_input,original_length = GenerateHahnSequence(pp, vary1, rng1, seq_name=seqname)
        else:
            raise Exception("Unknown pulse type.")
        
    elif is_AWG_parameter2:    
        rng2 = np.linspace(start2, stop2, num_points2)
        
        #keep track of the indices for each part of the sequence
        doNd = {val: 2*ind+1 for ind,val in enumerate(rng2)}
        def set_channel(val):
            ind = doNd[val]
            awg.current_step_ch1(ind)
            awg.current_step_ch2(ind)
            awg.current_step_ch3(ind)
            if vary2 == "C_ampl":
                pp.C_ampl = val
                for gate in cgp:
                    gate.update()
        
        setparam1 = vary1
        
        setparam2 = Parameter(vary2, label=PulseParameter.labels[vary2],
                         unit=PulseParameter.units[vary2],
                         set_cmd=set_channel)
        
        #create Experiment waveform
        #Rabi experiments:
        if pulse_type == "Rabi":
            seqname = f"do2dAWG_Rabi_{vary2}"
            seqx_input,original_length = GenerateRabiSequence(pp, vary2, rng2, seq_name=seqname)
        #Ramsey experiments:
        elif pulse_type == "Ramsey":
            seqname = f"do2dAWG_Ramsey_{vary2}"
            seqx_input,original_length = GenerateRamseySequence(pp, vary2, rng2, seq_name=seqname)
        elif pulse_type == "Hahn":
            seqname = f"do2dAWG_Hahn_{vary2}"
            seqx_input,original_length = GenerateHahnSequence(pp, vary2, rng2, seq_name=seqname)
        else:
            raise Exception("Unknown pulse type.")
        
    else:
        print("Stop! this is not a AWG scan!")

    if show_pulse:
        #plot sequence
        SequencePlotter(seqx_input, pp, original_length)

    #create seqx file
    seqx = awg.makeSEQXFile(*seqx_input)
    filename = seqname+'.seqx'

    #Transfer the seqx file
    awg.sendSEQXFile(seqx, filename)
    print(f"Sequence file of size {len(seqx)/1024/1024:.3f} MB was sent to AWG")

    #Load the seqx file
    awg.loadSEQXFile(filename)

    #Assign tracks from the sequence to the channels
    awg.ch1.setSequenceTrack(seqname, 1)
    awg.ch2.setSequenceTrack(seqname, 2)
    awg.ch3.setSequenceTrack(seqname, 3)

    #play!
    awg.ch1.state(1)
    awg.ch2.state(1)
    awg.ch3.state(1)
    awg.play()
    print("AWG is",awg.run_state())

    awg.wait_for_operation_to_complete()
    time.sleep(10)

    
    result = do2d(setparam1, start1, stop1, num_points1, delay1,
                setparam2, start2, stop2, num_points2, delay2,
                *param_meas,
                set_before_sweep = set_before_sweep,
                enter_actions = enter_actions,
                exit_actions = exit_actions,
                before_inner_actions = before_inner_actions,
                after_inner_actions = after_inner_actions,
                write_period = write_period,
                measurement_name = measurement_name,
                exp = exp,
                flush_columns = flush_columns,
                do_plot = do_plot,
                use_threads = use_threads,
                additional_setpoints = additional_setpoints,
                show_progress = show_progress,
                log_info = log_info)
    return result

    
def init_Rabi(pp: PulseParameter, awg: BaselAWG5204, cgp: CompensatedGateParameter, plot=True):
    """
    Method to initialize a Rabi experiment with the current parameters saved in pp
    """

    awg.sample_rate(pp.sampling_rate)
    awg.ch1.resolution(14)
    awg.ch2.resolution(14)
    awg.ch3.resolution(14)

    awg.ch1.awg_amplitude(1.5)
    awg.ch2.awg_amplitude(1.5)
    awg.ch3.awg_amplitude(1.5)

    seqname = f"initrabi"
    seqx_input,original_length = GenerateRabiSequence(pp, "", 0, seq_name=seqname)

    if plot:
        #plot sequence
        SequencePlotter(seqx_input, pp, original_length)

    seqx = awg.makeSEQXFile(*seqx_input)

    filename = seqname+'.seqx'
    # Step 4: Transfer the seqx file
    awg.sendSEQXFile(seqx, filename)

    print(f"Sequence file of size {len(seqx)/1024/1024:.3f} MB was sent to AWG")

    # Step 5: Load the seqx file
    awg.loadSEQXFile(filename)
    # Now the sequence should appear in the sequencelist, but it is not yet assigned to channels

    # Step 6: Assign tracks from the sequence to the channels
    # Unlike older/other AWG models, this can be done on a per-channel basis
    awg.ch1.setSequenceTrack(seqname, 1)
    awg.ch2.setSequenceTrack(seqname, 2)
    awg.ch3.setSequenceTrack(seqname, 3)

    awg.ch1.state(1)
    awg.ch2.state(1)
    awg.ch3.state(1)

    awg.play()
    print("AWG is",awg.run_state())

    for gate in cgp:
        gate.update()