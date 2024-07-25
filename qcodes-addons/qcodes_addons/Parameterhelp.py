import time
from typing import List, Optional

import numpy as np
from qcodes.instrument.parameter import (
    ManualParameter,
    MultiParameter,
    Parameter,
    ScaledParameter,
)
from qcodes.utils.validators import Numbers
from qcodes_addons.AWGhelp import PulseParameter


class GateParameter(Parameter):
    def __init__(self, param, name, value_range, unit: Optional[str]='V', 
                 scaling: Optional[float]=1, offset: Optional[float]=0):
        
        super().__init__(name=name, instrument=param.instrument, unit=unit,
                         vals=Numbers(value_range[0], value_range[1]))
    
        self.param = param
        self.scaling = scaling
        self.offset = offset
        self.vals = Numbers(value_range[0], value_range[1])
        
    def get_raw(self):
        return self.param.get()
    
    def set_raw(self,val):
        dacval = self.scaling*val+self.offset
        self.vals.validate(dacval)
        self.param.set(dacval)
        
    def range(self, value_range):
        self.vals = Numbers(value_range[0], value_range[1])
        

class VirtualGateParameter(Parameter):
    def __init__(self, name, params, set_scaling, 
                 offsets: Optional[List[float]]=None, 
                 get_scaling: Optional[float]=1 ):
        
        super().__init__(name=name, instrument=params[0].instrument, 
                         unit=params[0].unit)
        
        self.params = params
        self.set_scaling = set_scaling
        self.get_scaling = get_scaling
        
        if offsets is None:
            self.offsets = np.zeros(len(params))
        else:
            self.offsets = offsets
            
    def get_raw(self):
        return self.get_scaling*self.params[0].get()
        
    def set_raw(self, val):
        for i in range(len(self.params)):
            dacval = self.set_scaling[i]*val+self.offsets[i]
            self.params[i].set(dacval)
            
    def get_all(self):
        values = []
        keys = []
        for param in self.params:
            values.append(param.get())
            keys.append(param.name)
        return dict(zip(keys, values))   
        

        
        
        
        
class CompensatedGateParameter(Parameter):
    def __init__(self, param, name, value_range, pp: PulseParameter, unit: Optional[str]='V', 
                 scaling: Optional[float]=1):
        
        super().__init__(name=name, instrument=param.instrument, unit=unit,
                         vals=Numbers(value_range[0], value_range[1]))
    
        self.param = param
        self.scaling = scaling
        self.pp = pp
        self.vals = Numbers(value_range[0], value_range[1])
        self.gate_s = param.get()
        
    def get_raw(self):
        return self.gate_s #(self.params.get() - self.pp.C_ampl*self.pp.CP_correction_factor*0.5)/self.scaling
    
    def set_raw(self,val):
        self.gate_s = val
        #dacval = self.scaling*val+self.offset+self.pulseamp_s()*self.comp_factor*0.5
        dacval = self.scaling*val+self.pp.C_ampl*self.pp.CP_correction_factor*0.5
        self.vals.validate(dacval)
        self.param.set(dacval)
    
    def update(self):
        dacval = self.scaling*self.gate_s+self.pp.C_ampl*self.pp.CP_correction_factor*0.5
        self.vals.validate(dacval)
        self.param.set(dacval)
        
    
#     def range(self, value_range):
#         self.vals = Numbers(value_range[0], value_range[1])


        
        
        
        
# class CamplParameter(Parameter):
#     def __init__(self, param, name, value_range, pulseamp_s, gate_s, gate, unit: Optional[str]='V', 
#                  scaling: Optional[float]=1, offset: Optional[float]=0):
        
#         super().__init__(name=name, instrument=param.instrument, unit=unit,
#                          vals=Numbers(value_range[0], value_range[1]))
    
#         self.param = param
#         self.scaling = scaling
#         self.offset = offset
#         self.vals = Numbers(value_range[0], value_range[1])
#         self.pulseamp_s = pulseamp_s     #stored values of pulseamplitude and compensated gate voltage
#         self.gate_s = gate_s
#         self.gate = gate
        
#     def get_raw(self):
#         return self.pulseamp_s()
    
#     def set_raw(self,val):
#         self.pulseamp_s(val)
#         self.gate(self.gate_s())
#         dacval = self.scaling*val+self.offset
#         self.vals.validate(dacval)
#         self.param.set(dacval)
        
        
#     def range(self, value_range):
#         self.vals = Numbers(value_range[0], value_range[1])        

        
        
        
        
        
        
        
        
        
        
        
        
        
class MultiDAQParameter(MultiParameter):
    def __init__(self, params, name, gains: Optional[List[float]]=None):
        
        if type(params[0]) == ScaledParameter:
            self.names = [param.name for param in params]
            self.instr = params[0]._wrapped_parameter.root_instrument
        else:
            self.names = [param.label for param in params]
            self.instr = params[0].root_instrument

        self.units = [param.unit for param in params]
       
        super().__init__(name = name, names = self.names, units = self.units, 
                         shapes = ((),)*len(params), 
                         setpoints=((),)*len(params))
        
        self.daqchs = []
        self.gains = []
        for i in range(len(params)):
            if type(params[i]) == ScaledParameter:
                self.daqchs.append(int(params[i]._wrapped_parameter.label.replace('AI', '')))
                self.gains.append(params[i].division)
            else:
                self.daqchs.append(int(params[i].label.replace('AI', '')))
                if gains is not None:
                    self.gains.append(gains[i])
                else:
                    self.gains.append(1)
                           
    def get_raw(self):
        # read the values of all initialised DAQ channels
        vals = self.instr.channels.volt.get()
        return tuple(np.divide([vals[i] for i in self.daqchs], self.gains))
    

class ZILockinParameter(MultiParameter):
    def __init__(self, instr, params, name, names, gain, scaling, units):        
        
        super().__init__(name = name, names = names, units = units, 
                         shapes = ((),)*len(params), 
                         setpoints = ((),)*len(params))
        
        self._instr = instr
        self._params = params
        self.gain = gain
        self.scaling = scaling
        
    def get_raw(self):
        vals = []
        
        if isinstance(self.gain, ManualParameter):
            gain = self.gain()
        else: 
            gain = self.gain
            
        if isinstance(self.scaling, ManualParameter):
            scaling = self.scaling()
        else:
            scaling = self.scaling
        
        factor = float(scaling)/float(gain)
        result = self._instr.demods[0].sample()
        for param in self._params:
            if param.lower() == 'x':
                vals.append(result["x"]*factor)
            elif param.lower() == 'y':
                vals.append(result["y"]*factor)
            elif param.lower() == 'r':
                vals.append(np.sqrt(result["x"]*result["x"]+result["y"]*result["y"])*factor)
            elif param.lower() == 'phase':
                vals.append(np.angle(result["x"]+1j*result["y"], deg=True))
            elif param.lower() == 'pca':
                vals.append([result["x"], result["y"]])
            else:
                raise NotImplementedError("Only X, Y, R or Phase are valid inputs")
        if param.lower() == 'pca':
            vals = PCA_projection(list(np.moveaxis(vals, -1, 0)))
            return tuple(vals)
        else:
            return tuple(vals)

def PCA_projection(*arrays):
    arrays_copy = [np.copy(array) for array in arrays]
    for array in arrays_copy:
        array[np.isnan(array)] = np.mean(np.ma.masked_invalid(array))
        print(array)
    Z = np.stack(arrays_copy, axis=-1)
    shape= Z.shape
    u = np.mean(Z, axis=tuple(range(0, shape.__len__()-1)), keepdims=True)
    B = (Z-u).reshape(np.product(shape[0:-1]), shape[-1])
    C = np.einsum("ki, kj -> ij", B, B)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    arg_sorted = np.flip(eigen_values.argsort())
    eigen_vectors = eigen_vectors[:, arg_sorted]
    return np.einsum("ik, kj -> ij", B, eigen_vectors).reshape(shape)[..., 0]

class QMParameter(MultiParameter):
    def __init__(self, instr, params, name, names, gain, scaling, units):        
        
        super().__init__(name = name, names = names, units = units, 
                         shapes = ((),)*len(params), 
                         setpoints = ((),)*len(params))
        
        self._instr = instr
        self._params = params
        self.gain = gain
        self.scaling = scaling
        self._instr.run_exp()
        
    def get_raw(self):
        vals = []
        
        if isinstance(self.gain, ManualParameter):
            gain = self.gain()
        else: 
            gain = self.gain
            
        if isinstance(self.scaling, ManualParameter):
            scaling = self.scaling()
        else:
            scaling = self.scaling
        
        result = self._instr.get_res()
        for param in self._params:
            if param.lower() == 'x':
                vals.append(result[0])
            elif param.lower() == 'y':
                vals.append(result[1])
            elif param.lower() == 'r':
                vals.append(result[2])
            elif param.lower() == 'phase':
                vals.append(result[3])
            else:
                raise NotImplementedError("Only X, Y, R or Phase are valid inputs")
        return tuple(vals)
                    
                    
class AxisParameter(Parameter):
    def __init__(self, name, param):
        super().__init__(name = name, label = name, unit = param.position.unit, 
                         instrument = param.root_instrument)
        self.param = param
    
    def get_raw(self):
        return self.param.position.get()
        
    def set_raw(self, val):
        self.param.target_position.set(val)
        self.param.output.set('auto')
        self.param.enable_auto_move(relative=False)
        time.sleep(0.2)
        while self.param._get_status()["target"] == False:
            time.sleep(0.5)
        self.param.disable_auto_move()
        self.param.output.set('off')     
        
        





#new class from Simon G 3/11/2022
class SimonsVirtualGateParameter(Parameter):
    def __init__(self, matrix, virtual_voltages, gate_nr, name, params, 
                 offsets: Optional[List[float]]=None, 
                 get_scaling: Optional[float]=1 ):
        
        super().__init__(name=name, instrument=params[0].instrument, 
                         unit=params[0].unit)
        
        self.matrix = matrix
        self.vv = virtual_voltages
        self.gate_nr = gate_nr
        
        self.params = params
        self.get_scaling = get_scaling
        
        if offsets is None:
            self.offsets = np.zeros(len(params))
        else:
            # NOT IMPLEMENTED!
            self.offsets = np.zeros(len(params))#offsets
            
    def get_raw(self):
        return self.vv[self.gate_nr]
        
    def set_raw(self, val):
        self.vv[self.gate_nr] = val
        
        for i in range(len(self.params)):
            dacval = np.dot(self.matrix[i], self.vv)+self.offsets[i]
            self.params[i].set(dacval)
            
    def get_all(self):
        values = []
        keys = []
        for param in self.params:
            values.append(param.get())
            keys.append(param.name)
        return dict(zip(keys, values))   
