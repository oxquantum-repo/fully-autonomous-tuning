import zhinst.utils
import zhinst.ziPython as zi
import json, time
from packaging import version
from types import SimpleNamespace

class HDAWG_Core2x4():
    ziPython_min = version.parse("21.8.20085")
    labone_min = version.parse("21.8.20085")
    hdawg_fw_min = version.parse("67742")

    def __init__(self, daq, device, awg_index):
        """Configure the device. Mode of 2 channels grouped
        
        Parameters
        ----------
        daq : ziDAQServer 
            The DAQ connection
        device : str
            The serial of the HDAWG
        awg_index: int
            The index of the AWG core
        """

        self.daq = daq
        self.device = device
        self.awg_index = awg_index

        self.daq.connectDevice(device, '1gbe')

        self.reset_parameters()

        #Check versions
        self._check_versions()

        # Configure 2x4 mode
        self.daq.setString(f'/{self.device}/system/awg/channelgrouping', 'groups_of_4')

        # Setup AWG module
        self.awg_module = daq.awgModule()
        self.awg_module.set('device', device)
        self.awg_module.set('index', awg_index)
        # Execute commands
        self.awg_module.execute()

    def reset_parameters(self):
        #add a space for constants
        self.constants = SimpleNamespace()
        self.registers = SimpleNamespace()

    #Verify that all the components have the right version
    def _check_versions(self):
        ziPython_ver = version.parse(zi.__version__)
        
        labone_ver_raw = self.daq.getInt('/zi/about/revision')
        labone_ver = version.parse(f'{labone_ver_raw//10**7}.{labone_ver_raw//10**5%100}.{labone_ver_raw%10**5}')

        hdawg_fw_ver = version.parse(str(self.daq.getInt(f'/{self.device:s}/system/fwrevision')))

        if ziPython_ver < HDAWG_Core2x4.ziPython_min:
            raise Exception(f"The zhinst package needs to be updated to version {HDAWG_Core2x4.ziPython_min.public:s} or above!\n"
                            f"You can do that with 'pip install -U zhinst=={HDAWG_Core2x4.ziPython_min.public:s}'")
        if labone_ver < HDAWG_Core2x4.labone_min:
            raise Exception(f"The LabOne installation needs to be updated to version {HDAWG_Core2x4.labone_min.public:s} or above!\n"
                            "Please follow the instructions in the User Manual to perform the upgrade")
        if hdawg_fw_ver < HDAWG_Core2x4.hdawg_fw_min:
            raise Exception(f"The FW on device {self.device:s} needs to be updated!\n"
                            "Please follow the instructions in the User Manual to perform the FW upgrade")

    def config(self, program, cts=None, waves=None):
        """Configure the device. Mode of 8 channels grouped
        
        Parameters
        ----------
        index: int
            The index of the AWG core. In grouped mode, refers to the first one
        program: str
            The seqc program
        ct: dict
            The Command Table, as dictonary
        waves: list
            List of the waveforms
        """
        
        # Configure 2x4 mode
        awg_index = 0
        self.daq.setString(f'/{self.device}/system/awg/channelgrouping', 'groups_of_4')

        ## Configure AWG
        # Stop AWG
        self.daq.setInt(f'/{self.device}/awgs/{awg_index}/enable', 0)
        # Send sequence
        self.compile_seqc(program)

        #program each core
        for awg_index in range(1):
            # Run AWG program only once
            self.daq.setInt(f'/{self.device}/awgs/{awg_index}/single', 1)

            # Enable channel outputs
            self.daq.setInt(f'/{self.device}/sigouts/{awg_index*2}/on', 1)
            self.daq.setInt(f'/{self.device}/sigouts/{awg_index*2+1}/on', 1)


            #send AWG waves
            if waves is not None:
                for i, wave in enumerate(waves):
                    wave_raw = zhinst.utils.convert_awg_waveform(wave[0],wave[1])
                    self.daq.set(f'/{self.device}/awgs/{awg_index}/waveform/waves/{i}', wave_raw)

            #load the command table
            if cts is not None:
                for awg_index, ct in enumerate(cts):
                    #Create CT 
                    ct_all = {'header':{'version':'0.2'}, 'table':ct}
                    node = f"/{self.device:s}/awgs/{awg_index}/commandtable/data"
                    self.daq.setVector(node, json.dumps(ct_all))
                    
                    #debug print
                    #print(awg_index, daq.get(node,flat=True)[node][0]['vector'])
    def _setUserRegs(self):
        #if no registers are defined, skip this phase
        if not bool(self.registers.__dict__):
            return

        set_cmd = []
        for i,value in enumerate(self.registers.__dict__.values()):
            node = f'/{self.device:s}/awgs/{self.awg_index:d}/userregs/{i:d}'
            set_cmd.append((node, value))
        
        self.daq.set(set_cmd)

    def setHold(self, hold):
        self.daq.setInt(f'/{self.device:s}/awgs/{self.awg_index}/outputs/0/hold', hold)
        self.daq.setInt(f'/{self.device:s}/awgs/{self.awg_index}/outputs/1/hold', hold)
        self.daq.setInt(f'/{self.device:s}/awgs/{self.awg_index+1}/outputs/0/hold', hold)
        self.daq.setInt(f'/{self.device:s}/awgs/{self.awg_index+1}/outputs/1/hold', hold)
        
    def oscControl(self, enable):
        self.daq.setInt(f'/{self.device:s}/system/awg/oscillatorcontrol', enable)

    def frequency(self, freq, osc=0):
        self.daq.setDouble(f'/{self.device:s}/oscs/{osc}/freq', freq)

    def modulation(self, mode):
        self.daq.set(f'/{self.device:s}/awgs/{self.awg_index}/outputs/0/modulation/mode', mode)
        self.daq.set(f'/{self.device:s}/awgs/{self.awg_index}/outputs/1/modulation/mode', mode)

    def run(self, block=True):
        self._setUserRegs()
        node = f'/{self.device:s}/awgs/{self.awg_index}/enable'
        self.daq.syncSetInt(node, 1)
        if block:
            while(self.daq.getInt(node) == 1):
                time.sleep(0.005)

    def _const2seqc(self):
        """Transform the constants into
        valid seqc code
        """
        
        #if no constants are defined, return an empty string
        if not bool(self.constants.__dict__):
            return ""

        seqc = "//Constants definition\n"
            
        for name, value in self.constants.__dict__.items():
            seqc += f"const {name:s} = {value};\n"

        seqc += '\n'
        return seqc

    def _regs2seqc(self):
        """Transform the registers into
        valid seqc code
        """

        #if no registers are defined, return an empty string
        if not bool(self.registers.__dict__):
            return ""
        
        seqc = "//User registers\n"
            
        for i,name in enumerate(self.registers.__dict__.keys()):
            seqc += f"var {name:s} = getUserReg({i:d});\n"

        seqc += '\n'
        return seqc
        
    def compile_seqc(self, program):
        """Compile and send a sequence to the device
        
        Parameters
        ----------
        program: str
            The seqc program
        """

        #Add constants definitions
        constants = self._const2seqc()
        registers = self._regs2seqc()
        program = constants + registers + program

        # Compile program
        self.awg_module.set('compiler/sourcestring', program)
        while self.awg_module.getInt('compiler/status') == -1:
            time.sleep(0.1)
        if self.awg_module.getInt('compiler/status') == 1:
            msg = "Failed to compile program. Error message:\n"
            msg += self.awg_module.getString("compiler/statusstring")
            raise Exception(msg)
        if self.awg_module.getInt('compiler/status') == 2:
            msg = "Compilation successful with warnings. Warning message:\n"
            msg += self.awg_module.getString("compiler/statusstring")
            raise Warning(msg)

        # Upload program
        while (self.awg_module.getDouble('progress') < 1.0) and (self.awg_module.getInt('elf/status') != 1):
            time.sleep(0.5)
        if self.awg_module.getInt('elf/status') == 1:
            raise Exception("Failed to upload program.")
    
    def only_ct(self,cts):
        awg_index=0
        for awg_index, ct in enumerate(cts):
            #Create CT 
            ct_all = {'header':{'version':'0.2'}, 'table':ct}
            node = f"/{self.device:s}/awgs/{awg_index}/commandtable/data"
            # node = f"/{self.device:s}/awgs/0/commandtable/data"
            self.daq.setVector(node, json.dumps(ct_all))
            
def init_hdawg():
    
    #Define the parameters of the instrument
    dataserver_host = '127.0.0.1'
    dev_hd = 'dev8415'               #Device ID of the HDAWG
    awg_core = 0                     #AWG Core (equal to output_channel1/2)
    #Imports and initializations
    from zhinst.ziPython import ziDAQServer
    daq = ziDAQServer(dataserver_host, 8004, 6)
   
    # hdawg = HDAWG_Core(daq, dev_hd, awg_core)
    hdawg = HDAWG_Core2x4(daq, dev_hd, awg_core)
    
    
def preload_ramsey
    # create waveforms (to be sent to instrument)
    import numpy as np
    
    
    len_max_sample = int(np.ceil((300*2.4)/16) * 16)
    hdawg.constants.COULOMB_LEN_SAMPLE = len_max_sample
    tau_pi_sample = int((11.4*2.4))
    hdawg.constants.TAU_PI_SAMPLE = tau_pi_sample

    # tau_wait = variable
    tau_max = len_max_sample-2*tau_pi_sample
    print(f'maximum tau_wait is {tau_max/2.4} ns')

    waves = []
    waveforms_def = ""

    for tau_wait_sample in range(0,tau_max):
        small =np.concatenate((np.zeros(len_max_sample-2*tau_pi_sample-tau_wait_sample),np.ones(tau_pi_sample),np.zeros(tau_wait_sample),np.ones(tau_pi_sample)))
        small_180 =np.concatenate((np.zeros(len_max_sample-2*tau_pi_sample-tau_wait_sample),np.ones(tau_pi_sample),np.zeros(tau_wait_sample),(-1)*np.ones(tau_pi_sample))) # second pi/2 pulse     with opposite sign
        waves.append((small,small))
        # waves.append((small,small_180))  second pi/2 pulse with opposite sign
        waveforms_def += f"wave wave_{tau_wait_sample} = placeholder({len_max_sample});\nassignWaveIndex(1,wave_{tau_wait_sample},2,wave_{tau_wait_sample}, {tau_wait_sample});\n"

    ct12 = []
    for i in range(1,tau_max,1):
        entry = {
            'index': i,
            'waveform': {
                'index': i,
            },
            'amplitude0': {'value': 1},
        }
        ct12.append(entry)

    ct34 = []
    for i in range(tau_max):
        entry = {
            "index": i,
            "waveform": {
                "index": tau_max
            }
        }
        ct34.append(entry)

    seqc_program = f"""
    // waveforms
    {waveforms_def:s}

    // Coulomb pulse
    wave w_coulomb = ones({len_max_sample});
    assignWaveIndex(3,w_coulomb,4,w_coulomb,{tau_max});

    // short waveforms for hold
    wave wzero = zeros(32);
    wave wones = ones(32);

    // Sequence
    playZero(32);
    var i = getUserReg(0);
    repeat(10000){{
        playWave(1, wzero, 2, wzero, 3, wzero, 4, wzero);
        playZero(COULOMB_LEN_SAMPLE-32);
        executeTableEntry(i);
    }}
    """