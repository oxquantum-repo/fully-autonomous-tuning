from typing import Any, Callable, Dict, Optional, Union
import time

import qcodes.utils.validators as vals
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList

from ANC350Lib import ANC350LibActuatorType, ANC350v3Lib, ANC350v4Lib


class ANC350Axis(InstrumentChannel):
    '''
    A single axis of the ANC350 attocube controller

    Args:
        parent: the Instrument that the axis is attached to
        name:   axis name
        axis:   axis index (0,1,2)

    Attributes:
        position:   Get the current position of an axis
        frequency:  Set the frequency of the output signal
        amplitude:  Value for the piezo drive voltage 
        status:     Read axis status
        voltage: Sets the DC level on the voltage output when no sawtooth 
                 based motion and no feedback loop is active.
        target_position: Sets the target position for automatic motion 
                         (start_auto_move). For linear type actuators the
                         position unit is m, for goniometers and rotators it 
                         is degree.
        target_range: Defines the range around the target position where the 
                      target is considered to be reached.
        actuator: Selects the actuator to be used for the axis from actuator 
                  presets.
        actuator_name: Get the name of the currently selected actuator
        capacitance: Performs a measurement of the capacitance of the piezo 
                     motor and returns the result. If no motor is connected, 
                     the result will be 0. The function doesn't return before 
                     the measurement is complete; this will take a few seconds 
                     of time.
    '''

    def __init__(self, parent: "ANC350", name: str, axis: int):
        super().__init__(parent, name)

        self._axis = axis

        self.add_parameter("position",
                           label = "Position",
                           get_cmd = self._get_position,
                           set_cmd = None,
                           unit = "m or °")

        self.add_parameter("frequency",
                           label = "Frequency",
                           get_cmd = self._get_frequency,
                           set_cmd = self._set_frequency,
                           vals = vals.Ints(1, 5000),
                           unit = "Hz")

        self.add_parameter("amplitude",
                           label = "Amplitude",
                           get_cmd = self._get_amplitude,
                           set_cmd = self._set_amplitude,
                           vals = vals.Numbers(0, 70),
                           unit = "V")

        self.add_parameter("status",
                           label="Status",
                           get_cmd = self._get_status)

        self.add_parameter("target_position",
                           label = "Target Position",
                           get_cmd = None,
                           set_cmd = self._set_target_position,
                           vals = vals.Numbers(),
                           unit = "m or °")

        self.add_parameter("target_range",
                           label = "Target Range",
                           get_cmd = None,
                           set_cmd = self._set_target_range,
                           vals = vals.Numbers(),
                           unit = "m or °")

        self.add_parameter("actuator",
                           label = "Actuator",
                           get_cmd = None,
                           set_cmd = self._set_actuator,
                           vals = vals.Ints(0, 255))

        self.add_parameter("actuator_type",
                           label = "Actuator Type",
                           get_cmd = self._get_actuator_type)

        self.add_parameter("actuator_name",
                           label = "Actuator Name",
                           get_cmd = self._get_actuator_name)

        self.add_parameter("capacitance",
                           label = "Capacitance",
                           get_cmd = self._get_capacitance,
                           unit="F")

        if self._parent._version_no >= 4:
            voltage_get: Optional[Callable] = self._get_voltage
        else:
            voltage_get = None

        self.add_parameter("voltage",
                           label = "Voltage",
                           get_cmd = voltage_get,
                           set_cmd = self._set_voltage,
                           vals = vals.Numbers(0, 70),
                           unit = "V",
                           snapshot_exclude = True)

        self.add_parameter("output",
                           label = "Output",
                           val_mapping = {False: 0,
                                          True: 1,
                                          "off": 0,
                                          "on": 1,
                                          "auto": 2},
                           get_cmd = self._get_output,
                           set_cmd = self._set_output)

        # Set actual unit (either m or °) to positional parameters
        self._update_position_unit()

    # Version 3
    def single_step(self, backward: Optional[Union[bool, str, int]] = None) -> None:
        """
        Triggers a single step in desired direction.

        Args:
            backward: Step direction forward (False) or backward (True). 
                      Besides True/False, you can set the direction to 
                      "forward"/"backward" or +1/-1 (default: forward or False)
        """
        backward = self._map_direction_parameter(backward)

        self._parent._lib.start_single_step(self._parent._device_handle, 
                                            self._axis, 
                                            backward)

    
    def multiple_steps(self, steps: int) -> None:
        """
        Performs multiple steps. The direction depends on the sign 
                                 (+: forward, -: backward)

        Args:
            steps: Number of steps to move. The sign indicates the moving 
                   direction (+: forward,-: backward)
        """
        backward = (steps < 0)
        
        for i in range(abs(steps)):
            self.single_step(backward)

    
    def start_continuous_move(self, backward: Optional[Union[bool, str, int]] = None) -> None:
        """
        Starts continuous motion in forward or backward direction.
        Other kinds of motion are stopped.

        Args:
            backward: Step direction forward (False) or backward (True). 
                      Besides True/False, you can set the direction to 
                      "forward"/"backward" or +1/-1 (default: forward or False)
        """
        backward = self._map_direction_parameter(backward)

        self._parent._lib.start_continuous_move(self._parent._device_handle, 
                                                self._axis,
                                                True,
                                                backward)

    
    def stop_continuous_move(self) -> None:
        """
        Stops continuous motion in forward or backward direction.
        """
        self._parent._lib.start_continuous_move(self._parent._device_handle, 
                                                self._axis,
                                                False,
                                                False)

    
    _direction_mapping: Dict[Any, bool] = {"forward": False,
                                           "backward": True,
                                           +1: False,
                                           -1: True}

    @classmethod
    def _map_direction_parameter(cls, backward: Optional[Union[bool, str, int]]) -> bool:
        if backward is None:
            return False
        if backward in [False, True]:
            return bool(backward)

        if backward in cls._direction_mapping:
            return cls._direction_mapping[backward]

        raise ValueError(f"Unexpected value for argument `backward`. Allowed values are: "
                         f"{[None, False, True, *cls._direction_mapping.keys()]}")

    
    _relative_mapping: Dict[Any, bool] = {"absolute": True, "relative": False}

    
    def enable_auto_move(self, relative: Optional[Union[bool, str]] = None) -> None:
        """
        Enable automatic moving

        Args:
            relative: If the target position is to be interpreted absolute 
                      (False) or relative to the current position (True).
        """
        relative = self._map_relative_parameter(relative)

        self._parent._lib.start_auto_move(self._parent._device_handle, 
                                          self._axis, 
                                          True, 
                                          relative)

    
    def disable_auto_move(self) -> None:
        """
        Disable automatic moving
        """
        self._parent._lib.start_auto_move(self._parent._device_handle, 
                                          self._axis, 
                                          False, 
                                          False)

    
    @classmethod
    def _map_relative_parameter(cls, relative: Optional[Union[bool, str]]) -> bool:
        if relative is None:
            return False
        if relative in [False, True]:
            return relative

        if relative in cls._relative_mapping:
            return cls._relative_mapping[relative]
        
        allowed_values = ", ".join(cls._relative_mapping.keys())
        raise ValueError(f"Unexpected value for argument `relative`. Allowed values are: None, "
                         f"False, True, {allowed_values}")

    
    def _get_position(self) -> float:
        """
        Get the current axis position
        Returns: Current position in m (linear type actuators) or degrees 
                 (goniometers and rotators)
        """        
        return self._parent._lib.get_position(self._parent._device_handle, 
                                              self._axis)
        

    # def _set_position(self, position: float) -> None:
    #     """(EXPERIMENTAL FUNCTION)
    #     The axis moves to the given position with the target range that is set before.

    #     Args:
    #         position: The position the axis moves to
    #     """
    #     self._set_target_position(position)
    #     self._set_output(2)  # 2 = "auto"


    
    def _get_frequency(self) -> float:
        """
        Returns the frequency parameter of this axis.

        Returns:
            Frequency in Hertz [Hz], internal resolution is 1 Hz
        """
        return self._parent._lib.get_frequency(self._parent._device_handle, 
                                               self._axis)

    
    def _set_frequency(self, frequency: float) -> None:
        """
        Sets the frequency parameter for this axis

        Args:
            frequency (float): Frequency in Hertz [Hz], internal resolution is 1 Hz
        """
        frequency = int(round(frequency))
        self._parent._lib.set_frequency(self._parent._device_handle, 
                                        self._axis, 
                                        frequency)

    def _get_amplitude(self) -> float:
        """
        Returns the amplitude parameter of this axis.

        Returns:
            Amplitude in Volts [V]
        """
        return self._parent._lib.get_amplitude(self._parent._device_handle, 
                                               self._axis)

    def _set_amplitude(self, amplitude: float) -> None:
        """
        Sets the amplitude parameter for an axis

        Args:
            amplitude: Amplitude in Volts [V] (internal resolution is 1mV)
        """
        self._parent._lib.set_amplitude(self._parent._device_handle, 
                                        self._axis, 
                                        amplitude)

    def _get_status(self) -> Dict[str, bool]:
        """
        Reads status information about an axis

        Returns:
            A Dictionary containing the information about an axis:
                connected:  True, if the axis is connected to a sensor.
                enabled:    True, if the axis voltage output is enabled.
                moving:     True, if the axis is moving.
                target:     True, if the target is reached in automatic positioning.
                eot_fwd:    True, if end of travel detected in forward direction.
                eot_bwd:    True, if end of travel detected in backward direction.
                error:      True, if the axis' sensor is in error state.
        """
        keys = ("connected", "enabled", "moving", "target", "eot_fwd", "eot_bwf", "error")
        status = self._parent._lib.get_axis_status(self._parent._device_handle, self._axis)

        return dict(zip(keys, status))

    
    def _set_voltage(self, voltage: float) -> None:
        """
        Sets the DC level on the voltage output when no sawtooth based motion and no feedback loop
        is active.

        Args:
            voltage: DC output voltage in Volts [V], internal resolution is 1 mV
        """
        self._parent._lib.set_dc_voltage(self._parent._device_handle, self._axis, voltage)

    
    def _set_target_position(self, target: float) -> None:
        """
        Sets the target position for automatic motion.

        Args:
            target: Target position in m or degree.
        """
        self._parent._lib.set_target_position(self._parent._device_handle, 
                                              self._axis,
                                              target)
                                              
    def _set_target_range(self, target_range: float) -> None:
        """
        Sets the range around the target position where the target is 
        considered to be reached. For linear type actuators the position unit 
        is m, for goniometers and rotators it is degree.

        Args:
             target_range: Target range in m or degree
        """ 
        self._parent._lib.set_target_range(self._parent._device_handle, 
                                           self._axis,
                                           target_range)

    def _set_actuator(self, actuator: int) -> None:
        """
        Selects the actuator to be used for the axis from actuator presets. 
        And changes the unit of the position parameters.

        Args:
            actuator: Actuator selection (0..255)
        """
        self._parent._lib.select_actuator(self._parent._device_handle, 
                                          self._axis, 
                                          actuator)

        self._update_position_unit()

    def _update_position_unit(self) -> None:
        """
        Checks the current actuator type and sets the corresponding unit for 
        position parameters.
        """      
        if self._get_actuator_type() == ANC350LibActuatorType.Linear:
            unit = "m"
        else:
            unit = "°"
            
        self.position.unit = unit
        self.target_position.unit = unit
        self.target_range.unit = unit

    def _get_actuator_type(self) -> ANC350LibActuatorType:
        '''
        Get the type of the currently selected actuator
        Returns: Type of the actuator {0:linear, 1:goniometer, 2:rotator}
        '''
        return self._parent._lib.get_actuator_type(self._parent._device_handle,
                                                   self._axis)

    def _get_actuator_name(self) -> str:
        """
        Returns the name of the currently selected actuator
        Returns: Name of the actuator
        """
        return self._parent._lib.get_actuator_name(self._parent._device_handle, 
                                                   self._axis)

    def _get_capacitance(self) -> float:
        """
        Returns the motor capacitance
        Performs a measurement of the capacitance of the piezo motor and 
        returns the result. If no motor is connected, the result will be 0.
        The function doesn't return before the measurement is complete; this 
        will take a few seconds of time.

        Returns:
            Capacitance in Farad
        """
        return self._parent._lib.measure_capacitance(self._parent._device_handle, self._axis)

    def _set_output(self, enable: int) -> None:
        """
        Enables or disables the voltage output of this axis.

        Args:
            enable: Enable/disable voltage output:
                    - 0: disable
                    - 1: enable
                    - 2: enable until end of travel is detected
        """
        auto_off = False

        if enable == 0:
            enable = False
        elif enable == 1 or enable == 2:
            if enable == 2:  # enable, but automatically disable when end of travel is detected
                auto_off = True
            enable = True
        else:
            raise ValueError("enable")

        self._parent._lib.set_axis_output(self._parent._device_handle, self._axis, enable, auto_off)

    
    def _get_output(self) -> int:
        """Reads the voltage output status.

        Returns:
            1, if the axis voltage output is enabled. 0, if it is disabled.
        """
        return 1 if self._get_status()["enabled"] else 0

    
    # Version 4
    # ---------
    def _get_voltage(self) -> float:
        """
        Reads back the current DC level (only supported by library with version 4)

        Returns:
            DC output voltage in Volts [V]
        """
        return self._parent._lib.get_dc_voltage(self._parent._device_handle, self._axis)


class ANC350(Instrument):
    """
    QCoDeS driver for the attocube controller ANC350

    Args:
        name: the name of the instrument
        library: the file providing the dll wrappers
        serial: the serial number of the controller
    """

    def __init__(self, name: str, library: ANC350v3Lib, serial: str='undefined'):
        super().__init__(name)

        if isinstance(library, ANC350v4Lib):
            self._version_no = 4
        elif isinstance(library, ANC350v3Lib):
            self._version_no = 3
        else:
            raise NotImplementedError("Only v3 and v4 of the dll are supported")

        self._lib = library
        
        '''
        the discover function must not be called as long as any device is 
        connected. Thus, in order to be able to connect to more than one
        device, we first check with get_device_info if devices have already 
        been 'discovered'. get_device_info can be executed before discover,
        but will e.g. return '' for the serial number. It will connect to the
        ANC350 specified by its serial number. If not serial number is
        provided it will connect to the instrument with dev_no 0.
        '''
        
        if self._lib.get_device_info()[2] == "":
            # execute discover() only if no device is connected
            self._lib.discover()
        i = 0
        if serial != 'undefined':
            while True:
                # connect to the ANC350 specified by serial number
                if self._lib.get_device_info(dev_no=i)[2] == serial:
                    self._device_handle = self._lib.connect(dev_no=i)
                    self._device_no = i
                    # print(i)
                    break
                elif self._lib.get_device_info(dev_no=i)[2] == "":
                    raise Exception("ERROR: serial number not matching ")
                    break
                i += 1
        else: 
            # connect to ANC with dev_no=0
            self._device_handle = self._lib.connect() 
            self._device_no = 0
        
        axischannels = ChannelList(self, "ANC350Axis", ANC350Axis)
        for i in range(3):
            axis_name = f"ax{i+1}"
            axischannel = ANC350Axis(parent=self, name=axis_name, axis=i)
            axischannels.append(axischannel)
            self.add_submodule(axis_name, axischannel)
        axischannels.lock()
        self.add_submodule("axis_channels", axischannels)

        self.connect_message()
        
    
    def close(self) -> None:
        '''
        Closes the connection to the device. The device handle becomes invalid.
        '''
        self._lib.disconnect(self._device_handle)
        super().close()

    def save_params(self) -> None:
        '''
        Saves parameters to persistent flash memory in the device. They will 
        be present as defaults after the next power-on.
        '''
        self._lib.save_params(self._device_handle)

    def get_idn(self) -> Dict[str, Optional[str]]:
        '''
        Returns a dictionary with the device information
        '''
        serial = self._lib.get_device_info(self._device_no)[2]
        return {"vendor": "Attocube", 
                "model": "ANC350",
                "serial": serial, 
                "firmware": str(self._version_no)}
                
    def set_pos(self, axis: int, target: float, freq: float=None, amp: float=None, trange: float=None, rel: bool=False) -> None:
        if not isinstance(axis, int):
            raise TypeError
        ax_move = self._determine_axis(axis)
        
        if freq is not None:
            ax_move.frequency.set(freq)
        if amp is not None:
            ax_move.amplitude.set(amp)
        if trange is not None:
            ax_move.target_range.set(trange)
            
        ax_move.target_position.set(target)
        ax_move.output.set('auto')
        ax_move.enable_auto_move(relative=rel)
        time.sleep(0.2)
        while ax_move._get_status()["target"] == False:
            time.sleep(0.5)
        ax_move.disable_auto_move()
        ax_move.output.set('off')
    
    def start_cont_move(self, axis: int, backward: bool=False, freq: float=None, amp: float=None):
        if not isinstance(axis, int):
            raise TypeError
        
        ax_move = self._determine_axis(axis)
        
        if freq is not None:
            ax_move.frequency.set(freq)
        if amp is not None:
            ax_move.amplitude.set(amp)
        
        ax_move.output.set('auto')
        ax_move.start_continuous_move(backward)
    
    
    def stop_cont_move(self, axis):
        if not isinstance(axis, int):
            raise TypeError
        
        ax_move = self._determine_axis(axis)
        ax_move.stop_continuous_move()
        ax_move.output.set('off')
        
    
    def get_pos(self):
        return self.axis_channels[:].position.get()
        
    def _determine_axis(self, axis):
        if axis == 1:
            return self.ax1
        if axis == 2:
            return self.ax2
        if axis == 3:
            return self.ax3


if __name__ == '__main__':
    mydir = r"C:\Users\qcmos\qCMOS local\qcodes\addons"
    libfile = ANC350v4Lib(mydir+r"\anc350v4.dll")
    
    anc1 = ANC350("atto_controller1", library = libfile, serial="L 01 0493")
    
    time.sleep(0.5)
    print(anc1.ax1.position.get())
    print(anc1.ax1.capacitance.get())
    print(anc1.ax1.amplitude.get())
    
    # get position of all three axes
    print(anc1.get_pos())
    
    # continuous move
    anc1.start_cont_move(1, backward=True)
    time.sleep(2)
    anc1.stop_cont_move(1)
    
    # relative and absolute move
    # anc1.set_pos(1, -0.0001, 100, 30, rel=True)
    # anc1.set_pos(1, 0.0095, 100, 30, trange=1e-6, rel=False)
    print(anc1.ax1.position.get())

    
    anc2 = ANC350("atto_controller2", library = libfile, serial="L 01 0494")
    time.sleep(0.2)
    print(anc2.ax1.position.get())
    print(anc2.ax1.capacitance.get())
    print(anc2.ax1.amplitude.get())
    
    anc2.set_pos(1, 205, 100, 30, 0.01, rel=False)
    print(anc2.ax1.position.get())
    
    print(anc2.ax1.actuator_type.get())
    print(anc2.ax1.actuator_name.get())
    print(anc2.ax1.position.unit)
    print(anc2.ax1.target_range.get())

    anc1.close()
    anc2.close()