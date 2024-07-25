# Author: Simon Geyer
# Date:   12/01/2023
# Place:  Basel, Switzerland
import numpy as np
import matplotlib.pyplot as plt
import typing

class PulseParameter:
    """ 
    This class stores the pulse-related parameters used for the qubit waveform.
    When you add a parameter, don't forget to provide a default value, otherwise your previous code won't work.
    """
    def __init__(self,
            t_RO: float, 
            t_CB: float, 
            t_ramp: float,
            t_burst: float, 
            t_pi_2: float = 0,
            t_pi: float = 0,
            IQ_delay: float = 0,
            C_ampl: float = 0.5, 
            I_ampl: float = 0.35, 
            Q_ampl: float = 0.35,
            phi: float = 0, 
            artificial_oscillation: float = 50e6,
            f_SB: float = 0,
            sampling_rate: float = 2.5e9,
            CIQ_chs: typing.Tuple[int, int, int] = (1, 2, 3),
            f_lockin: float = 87.7777,
            t_wait = 0,
            CP_correction_factor = 0.1/1.5):
        """
        Args:
            t_RO:          Readout time (s)
            t_CB:          Coulomb pulse time (s)
            t_ramp:        Ramping time (s)
            t_burst:       Burst time (s)
            t_pi_2:        pi/2 pulse time
            t_pi_2:        pi pulse time
            IQ_delay:      IQ delay time (s)
            C_ampl:        Amplitude of the Coulomb pulse (V)
            I_ampl:        Amplitude of the I pulse = I0 (V)
            Q_ampl:        Amplitude of the Q pulse = Q0 (V)
            phi:           I_ampl = I0 * cos(phi) (degree)
            f_SB:          sideband frequenciy (Hz), 0 means no oscillations during the burst on I-Q channels
            sampling_rate: Sampling rate (# of samples / second)
            CIQ_chs:       Physical AWG channels for C, I, Q, in that order.
            f_lockin:      Frequency for chopping MW signal for the lockin measurement
            CP_correction_factor: Factor to correct the dc offset needed to compentsate the Coulomb pulse. Set this to triangle_splitting/C_ampl
            t_wait:        waiting time for Ramsey exp
        """
        self.t_RO = t_RO
        self.t_CB = t_CB
        self.t_ramp = t_ramp
        self.t_burst = t_burst
        self.t_pi_2 = t_pi_2
        self.t_pi = t_pi
        self.IQ_delay = IQ_delay
        self.I_ampl = I_ampl
        self.Q_ampl = Q_ampl
        self.C_ampl = C_ampl
        self.phi = phi
        self.artificial_oscillation = artificial_oscillation
        self.f_SB = f_SB
        self.sampling_rate = sampling_rate
        self.CIQ_chs = CIQ_chs
        self.f_lockin = f_lockin
        self.CP_correction_factor = CP_correction_factor
        
    # Add units and labels for convinient plotting.
    units = {
        "t_RO":          "s",
        "t_CB":          "s",
        "t_ramp":        "s",
        "t_burst":       "s",
        "t_pi_2":        "s",
        "t_pi":          "s",
        "IQ_delay":      "s",
        "C_ampl":        "V",
        "I_ampl":        "V",
        "Q_ampl":        "V",
        "phi":           "deg",
        "f_SB":          "Hz",
        "t_wait":        "s",
        "t_phi":         "s",
    }
    labels = {
        "t_RO":          "Readout time",
        "t_CB":          "Coulomb pulse time",
        "t_ramp":        "Ramp time",
        "t_burst":       "Burst time",
        "t_pi_2":        "Pi/2 time",
        "t_pi":          "Pi time",
        "IQ_delay":      "IQ delay time",
        "C_ampl":        "Coulomb pulse amplitude",
        "I_ampl":        "I pulse amplitude",
        "Q_ampl":        "V pulse amplitude",
        "phi":           "φ",
        "f_SB":          "Sideband frequency",
        "t_wait":        "Wait time",
        "t_phi":         "t phi",
    }

def pulseshape_to_array(sample_rate: float, pulseshape: tuple) -> list:
    """
    Transforms a tuple of tuples (duration, amplitude1, amplitude2) into a array of points. Could be expanded at some point to allow more complex segments than just ramps.
    """
    out = []
    for (d, a1, a2) in pulseshape:
        n = int(np.round(d*sample_rate))
        for i in range(n):
            out.append(a1+i/n*(a2-a1))
    return out
    
def create_CP(pp: PulseParameter) -> list:
    """
    create a coulomb pulse with the parameters found in pp
    """
    t_RO = pp.t_RO
    t_CB = pp.t_CB
    t_ramp = pp.t_ramp
    C_ampl = pp.C_ampl
    sampling_rate = pp.sampling_rate
    
    #total number of points to be expected
    n_points = int(np.ceil((t_RO+t_CB)*sampling_rate))
    
    #define Coulomb pulse without ramp (should be added here)
    if t_ramp == 0:
        pulseshape = ((t_RO,-C_ampl/2,-C_ampl/2),
                      (t_CB,C_ampl/2,C_ampl/2))
    else:
        pulseshape = ((t_ramp/2, 0, -C_ampl/2),
                      (t_RO-t_ramp,-C_ampl/2,-C_ampl/2),
                      (t_ramp, -C_ampl/2, C_ampl/2),
                      (t_CB-t_ramp,C_ampl/2,C_ampl/2),
                      (t_ramp/2,C_ampl/2,0))
    out = pulseshape_to_array(sampling_rate, pulseshape)
    
    #check length
    d = len(out) - n_points
    if d > 0:
        out = out[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out = out+l
    return out
    

def create_Rabi_IQ(pp: PulseParameter) -> list:
    """
    create  IQ channel for Rabi experiment with parameters saved in pp
    """
    t_IQstart = pp.t_RO+pp.t_CB-pp.t_burst
    
    #total number of points to be expected
    n_points = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
    
    
    #IQ pulseshape
    pulseshape = ((t_IQstart,0,0),(pp.t_burst,pp.I_ampl,pp.I_ampl))
    out = pulseshape_to_array(pp.sampling_rate, pulseshape)
    
    #wrap array because of IQdelay and t_ramp
    n_points_delay = int(np.round((pp.IQ_delay+pp.t_ramp/2)*pp.sampling_rate))
    n_points_delay = np.mod(n_points_delay, n_points)
    out = out[n_points_delay:-1] + out[0:n_points_delay]
        
    #check if length is correct
    d = len(out) - n_points
    if d > 0:
        out = out[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out = out+l
       
    #calc I and Q channels
    if pp.f_SB==0:
        outI = out
        outQ = list(np.asarray(out)*0)
    else:
        outI = [o*np.cos(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out)]
        outQ = [o*np.sin(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out)]
    
    return outI, outQ
    

def GenerateRabiSequence(pp: PulseParameter, vary: str, rng: list, seq_name: str='') -> list:
    """
    General function to ceate sequence of Rabi pulses where 1 parameter is varied
    """
    

#Init Rabi, i.e. no vary:
    if vary == "":
        CP_array = create_CP(pp)
        I_array, Q_array = create_Rabi_IQ(pp)
                
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        if len(I_array) != desired_length or len(Q_array) != desired_length:
            print("IQ pulse array has wrong length!")
        
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        I_array = I_array*fill_to_min_length
        Q_array = Q_array*fill_to_min_length
                
        #create marker channel signals
        m_low = np.zeros(len(Q_array))
        m_high = np.ones(len(Q_array))
        
        #next I choose the lockin-cycle as simple RF on/off
        wfm_ch1_n1 = np.array([CP_array, m_low, m_low])
        wfm_ch1_n2 = np.array([CP_array, m_low, m_low])
        wfm_ch2_n1 = np.array([I_array, m_high, m_low])
        wfm_ch2_n2 = np.array([list(np.asarray(I_array)*0), m_low, m_low])
        wfm_ch3_n1 = np.array([Q_array, m_high, m_low])
        wfm_ch3_n2 = np.array([list(np.asarray(Q_array)*0), m_low, m_low])
        
        trig_waits = [0, 0]  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0, 0]  # 0 corresponds to infinite
        event_jumps = [1, 2] # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [2, 1]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0, 1]  # 0 means next
        
        seqname = seq_name
        
        wfms = [[wfm_ch1_n1, wfm_ch1_n2], [wfm_ch2_n1, wfm_ch2_n2],[wfm_ch3_n1, wfm_ch3_n2]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length   
    
    
# Vary the burst time of the Rabi
    elif vary == "t_burst":
        n = len(rng)

        CP_array = create_CP(pp)
        I_array, Q_array = [],[]

        for r in rng:
            pp.t_burst = r
            I_array_temp, Q_array_temp = create_Rabi_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)

        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")

        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length

        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))

        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))

        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next


        seqname = seq_name
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5

        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length
    
#vary the coulomb pulse amplitude
    elif vary == "C_ampl":
        n = len(rng)
        
        CP_array = []
        I_array, Q_array = create_Rabi_IQ(pp)
        
        for r in rng:
            pp.C_ampl = r
            CP_array_temp = create_CP(pp)
            CP_array.append(CP_array_temp)
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(I_array) != desired_length or len(Q_array) != desired_length:
            print("IQ pulse array has wrong length!")
        for c in CP_array:
            if len(c) != desired_length:
                print("C pulse array has wrong length!")
        
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(I_array))))
        I_array = I_array*fill_to_min_length
        Q_array = Q_array*fill_to_min_length
        for ind in range(len(CP_array)):
            CP_array[ind] = CP_array[ind]*fill_to_min_length
                
        #create marker channel signals
        m_low = np.zeros(len(I_array))
        m_high = np.ones(len(I_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array[i], m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array[i], m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array, m_high, m_low]))
            wfm_ch2_n.append(np.array([list(np.asarray(I_array)*0), m_low, m_low]))
            wfm_ch3_n.append(np.array([Q_array, m_high, m_low]))
            wfm_ch3_n.append(np.array([list(np.asarray(Q_array)*0), m_low, m_low]))
        
        
        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] #list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n)#[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next
        
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length  
#TODO: add other variables to vary, e.g. the sideband frequency     
    else:
        print("Unkown vary argument")
        
        
def create_Ramsey_IQ(pp: PulseParameter) -> list:
    """
    create IQ channel for Ramsey sequence with parameters found in pp
    """
    t_IQstart = pp.t_RO+pp.t_CB-pp.t_wait-2*pp.t_pi_2
    
    #total number of points to be expected
    n_points = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
    
    
    #IQ pulseshape
    pulseshape_pulse1 = ((t_IQstart,0,0),(pp.t_pi_2,pp.I_ampl,pp.I_ampl),(pp.t_wait,0,0),(pp.t_pi_2,0,0))
    out1 = pulseshape_to_array(pp.sampling_rate, pulseshape_pulse1)
    
    #wrap array because of IQdelay and t_ramp
    n_points_delay = int(np.round((pp.IQ_delay+pp.t_ramp/2)*pp.sampling_rate))
    n_points_delay = np.mod(n_points_delay, n_points)
    out1 = out1[n_points_delay:-1] + out1[0:n_points_delay]
        
    #check if length is correct
    d = len(out1) - n_points
    if d > 0:
        out1 = out1[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out1 = out1+l
        
    #IQ pulseshape for second pulse, allows different phases for both pulses
    pulseshape_pulse2 = ((t_IQstart,0,0),(pp.t_pi_2,0,0),(pp.t_wait,0,0),(pp.t_pi_2,pp.I_ampl,pp.I_ampl))
    out2 = pulseshape_to_array(pp.sampling_rate, pulseshape_pulse2)
    
    #wrap array because of IQdelay and t_ramp
    n_points_delay = int(np.round((pp.IQ_delay+pp.t_ramp/2)*pp.sampling_rate))
    n_points_delay = np.mod(n_points_delay, n_points)
    out2 = out2[n_points_delay:-1] + out2[0:n_points_delay]
        
    #check if length is correct
    d = len(out2) - n_points
    if d > 0:
        out2 = out2[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out2 = out2+l
       
    #calc I and Q channels
    outI1 = [o*np.cos(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out1)]
    outQ1 = [o*np.sin(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out1)]
    outI2 = [o*np.cos(2*np.pi*n/pp.sampling_rate*pp.f_SB+2*np.pi*pp.phi/360) for n,o in enumerate(out2)]
    outQ2 = [o*np.sin(2*np.pi*n/pp.sampling_rate*pp.f_SB+2*np.pi*pp.phi/360) for n,o in enumerate(out2)]
    
    outI=list(np.asarray(outI1)+np.asarray(outI2))
    outQ=list(np.asarray(outQ1)+np.asarray(outQ2))
    return outI, outQ
        
def GenerateRamseySequence(pp: PulseParameter, vary: str, rng: list, seq_name: str='') -> list:
    """
    General function to ceate sequence of Ramsey pulses with 1 varying parameter
    """
    phase_flip_projection = True
#TODO: implement init_Ramsey
    if vary == "":
        print("Init_Ramsey is not implemented yet")
        return 0

#Change wait time of Ramsey experiment
    elif vary == "t_wait":
        
        n = len(rng)
        
        CP_array = create_CP(pp)
        I_array, Q_array = [],[]
        I_array2, Q_array2 = [],[]
        
        for r in rng:
            pp.t_wait = r
            I_array_temp, Q_array_temp = create_Ramsey_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)
        
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")
                
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length
            
        if phase_flip_projection:
            phi_temp = pp.phi
            pp.phi = phi_temp+180
            for r in rng:
                pp.t_wait = r
                I_array_temp2, Q_array_temp2 = create_Ramsey_IQ(pp)
                I_array2.append(I_array_temp2)
                Q_array2.append(Q_array_temp2)
                
            pp.phi = phi_temp
                
            #check point length for consistency
            desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
            for i,q in zip(I_array2, Q_array2):
                if len(i) != desired_length or len(q) != desired_length:
                    print("IQ pulse array has wrong length!")

            #repeat the sequence until minimum 2400 points are reached
            for ind in range(len(I_array2)):
                I_array2[ind] = I_array2[ind]*fill_to_min_length
                Q_array2[ind] = Q_array2[ind]*fill_to_min_length        
                
                
        
        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            if phase_flip_projection:
                wfm_ch2_n.append(np.array([I_array2[i], m_low, m_low]))
                wfm_ch3_n.append(np.array([Q_array2[i], m_low, m_low]))
            else:
                wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
                wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))
        

        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next


            
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length

#Change wait time of Ramsey experiment
    elif vary == "t_phi":
        
        n = len(rng)
        
        CP_array = create_CP(pp)
        I_array, Q_array = [],[]
        I_array2, Q_array2 = [],[]
                
        for r in rng:
            pp.t_wait = r
            pp.phi = 360*pp.artificial_oscillation*r
            I_array_temp, Q_array_temp = create_Ramsey_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)
        pp.phi=0
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")
                
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length
        
        if phase_flip_projection:
            phi_temp = pp.phi
            for r in rng:
                pp.t_wait = r
                pp.phi = 360*pp.artificial_oscillation*r+180
                I_array_temp2, Q_array_temp2 = create_Ramsey_IQ(pp)
                I_array2.append(I_array_temp2)
                Q_array2.append(Q_array_temp2)
                
            pp.phi = phi_temp
                
            #check point length for consistency
            desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
            for i,q in zip(I_array2, Q_array2):
                if len(i) != desired_length or len(q) != desired_length:
                    print("IQ pulse array has wrong length!")

            #repeat the sequence until minimum 2400 points are reached
            for ind in range(len(I_array2)):
                I_array2[ind] = I_array2[ind]*fill_to_min_length
                Q_array2[ind] = Q_array2[ind]*fill_to_min_length        
                
        pp.phi=0
        
        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            if phase_flip_projection:
                wfm_ch2_n.append(np.array([I_array2[i], m_low, m_low]))
                wfm_ch3_n.append(np.array([Q_array2[i], m_low, m_low]))
            else:
                wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
                wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))
        
        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next

        
            
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length
    
#Vary the t_pi_2 time:
    elif vary == "t_pi_2":
        
        n = len(rng)
        
        CP_array = create_CP(pp)
        I_array, Q_array = [],[]
        
        for r in rng:
            pp.t_pi_2 = r
            I_array_temp, Q_array_temp = create_Ramsey_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")
                
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length
        
        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))
        
        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next

        
            
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length
    
    else:
        print("Unkown vary argument")
        
        
def create_Hahn_IQ(pp: PulseParameter) -> list:
    """
    create IQ channel for Hahn sequence with parameters found in pp
    """
    t_IQstart = pp.t_RO+pp.t_CB-pp.t_wait-2*pp.t_pi_2-pp.t_pi
    
    #total number of points to be expected
    n_points = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
    
    
    #IQ pulseshape
    pulseshape_pulse1 = ((t_IQstart,0,0),(pp.t_pi_2,pp.I_ampl,pp.I_ampl),(pp.t_wait/2,0,0),(pp.t_pi,pp.I_ampl,pp.I_ampl),(pp.t_wait/2,0,0),(pp.t_pi_2,0,0))
    out1 = pulseshape_to_array(pp.sampling_rate, pulseshape_pulse1)
    
    #wrap array because of IQdelay and t_ramp
    n_points_delay = int(np.round((pp.IQ_delay+pp.t_ramp/2)*pp.sampling_rate))
    n_points_delay = np.mod(n_points_delay, n_points)
    out1 = out1[n_points_delay:-1] + out1[0:n_points_delay]
        
    #check if length is correct
    d = len(out1) - n_points
    if d > 0:
        out1 = out1[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out1 = out1+l
        
    #IQ pulseshape for second t_pi_2 pulse, allows different phases for final pulse
    pulseshape_pulse2 = ((t_IQstart,0,0),(pp.t_pi_2,0,0),(pp.t_wait/2,0,0),(pp.t_pi,0,0),(pp.t_wait/2,0,0),(pp.t_pi_2,pp.I_ampl,pp.I_ampl))
    out2 = pulseshape_to_array(pp.sampling_rate, pulseshape_pulse2)
    
    #wrap array because of IQdelay and t_ramp
    n_points_delay = int(np.round((pp.IQ_delay+pp.t_ramp/2)*pp.sampling_rate))
    n_points_delay = np.mod(n_points_delay, n_points)
    out2 = out2[n_points_delay:-1] + out2[0:n_points_delay]
        
    #check if length is correct
    d = len(out2) - n_points
    if d > 0:
        out2 = out2[0:n_points]
    if d < 0:
        l = [0]*abs(d)
        out2 = out2+l
       
    #calc I and Q channels
    outI1 = [o*np.cos(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out1)]
    outQ1 = [o*np.sin(2*np.pi*n/pp.sampling_rate*pp.f_SB) for n,o in enumerate(out1)]
    outI2 = [o*np.cos(2*np.pi*n/pp.sampling_rate*pp.f_SB+2*np.pi*pp.phi/360) for n,o in enumerate(out2)]
    outQ2 = [o*np.sin(2*np.pi*n/pp.sampling_rate*pp.f_SB+2*np.pi*pp.phi/360) for n,o in enumerate(out2)]
    
    outI=list(np.asarray(outI1)+np.asarray(outI2))
    outQ=list(np.asarray(outQ1)+np.asarray(outQ2))
    return outI, outQ
        
def GenerateHahnSequence(pp: PulseParameter, vary: str, rng: list, seq_name: str='') -> list:
    """
    General function to ceate sequence of Hahn-Echo pulses with 1 varying parameter
    """
#TODO: implement init_Hahn
    if vary == "":
        print("Init_Hahn is not implemented yet")
        return 0

#Change wait time of Hahn experiment
    elif vary == "t_wait":
        
        n = len(rng)
        
        CP_array = create_CP(pp)
        I_array, Q_array = [],[]
        
        for r in rng:
            pp.t_wait = r
            I_array_temp, Q_array_temp = create_Hahn_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")
                
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length
        
        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))
        
        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next
        
            
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length
    
#Change wait time of Hahn experiment
    elif vary == "t_phi":
        
        n = len(rng)
        
        CP_array = create_CP(pp)
        I_array, Q_array = [],[]
                
        for r in rng:
            pp.t_wait = r
            pp.phi = 360*pp.artificial_oscillation*r
            I_array_temp, Q_array_temp = create_Hahn_IQ(pp)
            I_array.append(I_array_temp)
            Q_array.append(Q_array_temp)
        pp.phi=0
        
        #check point length for consistency
        desired_length = int(np.ceil((pp.t_RO+pp.t_CB)*pp.sampling_rate))
        if len(CP_array) != desired_length:
            print("Coulomb pulse array has wrong length!")
        for i,q in zip(I_array, Q_array):
            if len(i) != desired_length or len(q) != desired_length:
                print("IQ pulse array has wrong length!")
                
        #repeat the sequence until minimum 2400 points are reached
        fill_to_min_length = int(np.ceil(2400/(len(CP_array))))
        CP_array = CP_array*fill_to_min_length
        for ind in range(len(I_array)):
            I_array[ind] = I_array[ind]*fill_to_min_length
            Q_array[ind] = Q_array[ind]*fill_to_min_length
            
        #create marker channel signals
        m_low = np.zeros(len(CP_array))
        m_high = np.ones(len(CP_array))
        
        wfm_ch1_n = []
        wfm_ch2_n = []
        wfm_ch3_n = []
        for i in range(n):
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch1_n.append(np.array([CP_array, m_low, m_low]))
            wfm_ch2_n.append(np.array([I_array[i], m_high, m_low]))
            wfm_ch2_n.append(np.array([list(np.asarray(I_array[i])*0), m_low, m_low]))
            wfm_ch3_n.append(np.array([Q_array[i], m_high, m_low]))
            wfm_ch3_n.append(np.array([list(np.asarray(Q_array[i])*0), m_low, m_low]))
        
        trig_waits = [0]*(2*n)  # 0: off, 1: trigA, 2: trigB, 3: EXT
        nreps = [0]*(2*n)  # 0 corresponds to infinite
        event_jumps = [1,2]*n # 0: off, 1: trigA, 2: trigB, 3: EXT
        event_jump_to = [item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist]#list(np.arange(3,2*n+1))+[1,2]  # irrelevant if event-jump is 0, else the sequence pos. to jump to
        go_to = [0]*(2*n) #[item for sublist in [[2*i,2*i-1] for i in range(1,n+1)] for item in sublist] # 0 means next

        
            
        seqname = seq_name
        
        wfms = [[*wfm_ch1_n], [*wfm_ch2_n], [*wfm_ch3_n]]

        ch1_amp, ch2_amp, ch3_amp = 1.5, 1.5, 1.5
        
        seqx_input = [trig_waits,
                                nreps,
                                event_jumps,
                                event_jump_to,
                                go_to,
                                wfms,
                                [ch1_amp, ch2_amp, ch3_amp],
                                seqname]
        return seqx_input, desired_length
    
    else:
        print("Unkown vary argument")

def SequencePlotter(seqx_input: list, pp: PulseParameter, original_length: int):
    
    trig_waits, nreps, event_jumps, event_jump_to, go_to, wfms, ch_ampls, seqname = seqx_input
    
    #plot step 1 and step n
    n = len(trig_waits)-2
    fig, axs = plt.subplots(2, 2, sharey='row')
    t = 1e9*np.arange(original_length)/pp.sampling_rate
    axs[0,0].plot(t, wfms[0][0][0][0:original_length])
    axs[1,0].plot(t, wfms[1][0][0][0:original_length])
    axs[1,0].plot(t, wfms[2][0][0][0:original_length], c="C1")
    axs[0,0].plot(t, wfms[0][1][0][0:original_length],c="C0",ls="dashed")
    axs[1,0].plot(t, wfms[2][1][0][0:original_length], c="C1",ls="dashed")
    axs[1,0].plot(t, wfms[1][1][0][0:original_length],c="C0",ls="dashed")
    axs[0,0].title.set_text("step 1")
    axs[0,0].set_ylabel("C_ampl (V)")
    axs[1,0].set_ylabel("IQ ampl (V)")
    axs[1,0].set_xlabel("time (ns)")

    axs[0,1].plot(t, wfms[0][n][0][0:original_length])
    axs[1,1].plot(t, wfms[1][n][0][0:original_length])
    axs[1,1].plot(t, wfms[2][n][0][0:original_length], c="C1")
    axs[0,1].plot(t, wfms[0][n+1][0][0:original_length],c="C0",ls="dashed")
    axs[1,1].plot(t, wfms[2][n+1][0][0:original_length],c="C1",ls="dashed")
    axs[1,1].plot(t, wfms[1][n+1][0][0:original_length],c="C0",ls="dashed")
    axs[0,1].title.set_text("step "+str(int(np.ceil(n/2+1))))
    axs[1,0].set_xlabel("time (ns)")
    plt.show()