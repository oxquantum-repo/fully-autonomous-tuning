from .AWG5204 import AWG5204
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
from typing import Any, Dict, List, Optional, Sequence, Union
import xml.etree.ElementTree as ET
import numpy as np
import time
import datetime as dt
import io
import zipfile as zf
from broadbean.sequence import InvalidForgedSequenceError, fs_schema
import logging
log = logging.getLogger(__name__)


class BaselAWG5204(AWG5204): 
    JUMP_TIMING = 'JumpEnd' # 'JumpImmed'
         
    def __init__(self, name: str, address: str,
                 timeout: float = 10, **kwargs: Any) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).
        """
        super().__init__(name, address, timeout=timeout, **kwargs)
        
        self.add_parameter('current_step_ch1',
                           label=(f'Current step for channel 1'),
                           set_cmd=f"SOURce1:JUMP:FORCe {{}}"
                           #get_cmd=f"SOURce{self.channel}:SCSTep?",
                           #get_parser=float
                          )
        
        self.add_parameter('current_step_ch2',
                           label=(f'Current step for channel 2'),
                           set_cmd=f"SOURce2:JUMP:FORCe {{}}"
                           #get_cmd=f"SOURce{self.channel}:SCSTep?",
                           #get_parser=float
                          )
        self.add_parameter('current_step_ch3',
                           label=(f'Current step for channel 3'),
                           set_cmd=f"SOURce3:JUMP:FORCe {{}}"
                           #get_cmd=f"SOURce{self.channel}:SCSTep?",
                           #get_parser=float
                          )
    
    #override this method and implement a proper check for the array length    
    @staticmethod
    def make_SEQX_from_forged_sequence(
            seq: Dict[int, Dict[Any, Any]],
            original_length: int,
            amplitudes: List[float],
            seqname: str,
            channel_mapping: Optional[Dict[Union[str, int], int]] = None
    ) -> bytes:
        """
        Make a .seqx from a forged broadbean sequence.
        Supports subsequences.
        Args:
            seq: The output of broadbean's Sequence.forge()
            amplitudes: A list of the AWG channels' voltage amplitudes.
                The first entry is ch1 etc.
            channel_mapping: A mapping from what the channel is called
                in the broadbean sequence to the integer describing the
                physical channel it should be assigned to.
            seqname: The name that the sequence will have in the AWG's
                sequence list. Used for loading the sequence.
        Returns:
            The binary .seqx file contents. Can be sent directly to the
                instrument or saved on disk.
        """

        try:
            fs_schema.validate(seq)
        except Exception as e:
            raise InvalidForgedSequenceError(e)

        chan_list: List[Union[str, int]] = []
        for pos1 in seq.keys():
            for pos2 in seq[pos1]['content'].keys():
                for ch in seq[pos1]['content'][pos2]['data'].keys():
                    if ch not in chan_list:
                        chan_list.append(ch)

        if channel_mapping is None:
            channel_mapping = {ch: ch_ind+1
                               for ch_ind, ch in enumerate(chan_list)}

        if len(set(chan_list)) != len(amplitudes):
            raise ValueError('Incorrect number of amplitudes provided.')

        if set(chan_list) != set(channel_mapping.keys()):
            raise ValueError(f'Invalid channel_mapping. The sequence has '
                             f'channels {set(chan_list)}, but the '
                             'channel_mapping maps from the channels '
                             f'{set(channel_mapping.keys())}')

        if set(channel_mapping.values()) != set(range(1, 1+len(chan_list))):
            raise ValueError('Invalid channel_mapping. Must map onto '
                             f'{list(range(1, 1+len(chan_list)))}')

        ##########
        # STEP 1:
        # Make all .wfmx files

        wfmx_files: List[bytes] = []
        wfmx_filenames: List[str] = []

        for pos1 in seq.keys():
            for pos2 in seq[pos1]['content'].keys():
                for ch, data in seq[pos1]['content'][pos2]['data'].items():
                    wfm = data['wfm']
                    print(wfm.shape)

                    markerdata = []
                    
                    
                    #check for proper length:
                    d = len(wfm) - original_length
                    if d > 0:
                        for mkey in ['m1', 'm2', 'm3', 'm4']:
                            if mkey in data.keys():
                                markerdata.append(data.get(mkey)[0:original_length])
                        wfm = wfm[0:original_length]
                    if d < 0:
                        l = [0]*abs(d)
                        wfm = np.ndarray(shape=(2500,),buffer=np.asarray(list(wfm)+l))
                        for mkey in ['m1', 'm2', 'm3', 'm4']:
                            if mkey in data.keys():
                                markerdata.append(np.ndarray(shape=(2500,),buffer=np.asarray(list(data.get(mkey))+l)))
                    if d == 0:
                        for mkey in ['m1', 'm2', 'm3', 'm4']:
                            if mkey in data.keys():
                                markerdata.append(data.get(mkey))
                    
                    print(len(wfm))
                    print("marker: "+str(len(markerdata[0])))
                    
                    wfm_data = np.stack((wfm, *markerdata))

                    awgchan = channel_mapping[ch]
                    wfmx = AWG70000A.makeWFMXFile(wfm_data,
                                                  amplitudes[awgchan-1])
                    wfmx_files.append(wfmx)
                    wfmx_filenames.append(f'wfm_{pos1}_{pos2}_{awgchan}')

        ##########
        # STEP 2:
        # Make all subsequence .sml files

        log.debug(f'Waveforms done: {wfmx_filenames}')

        subseqsml_files: List[str] = []
        subseqsml_filenames: List[str] = []

        for pos1 in seq.keys():
            if seq[pos1]['type'] == 'subsequence':

                ss_wfm_names: List[List[str]] = []

                # we need to "flatten" all the individual dicts of element
                # sequence options into one dict of lists of sequencing options
                # and we must also provide default values if nothing
                # is specified
                seqings: List[Dict[str, int]] = []
                for pos2 in (seq[pos1]['content'].keys()):
                    pos_seqs = seq[pos1]['content'][pos2]['sequencing']
                    pos_seqs['twait'] = pos_seqs.get('twait', 0)
                    pos_seqs['nrep'] = pos_seqs.get('nrep', 1)
                    pos_seqs['jump_input'] = pos_seqs.get('jump_input', 0)
                    pos_seqs['jump_target'] = pos_seqs.get('jump_target', 0)
                    pos_seqs['goto'] = pos_seqs.get('goto', 0)
                    seqings.append(pos_seqs)

                    ss_wfm_names.append([n for n in wfmx_filenames
                                         if f'wfm_{pos1}_{pos2}' in n])

                seqing = {k: [d[k] for d in seqings]
                          for k in seqings[0].keys()}

                subseqname = f'subsequence_{pos1}'

                log.debug(f'Subsequence waveform names: {ss_wfm_names}')

                subseqsml = BaselAWG5204._makeSMLFile(trig_waits=seqing['twait'],
                                                   nreps=seqing['nrep'],
                                                   event_jumps=seqing['jump_input'],
                                                   event_jump_to=seqing['jump_target'],
                                                   go_to=seqing['goto'],
                                                   elem_names=ss_wfm_names,
                                                   seqname=subseqname,
                                                   chans=len(channel_mapping))

                subseqsml_files.append(subseqsml)
                subseqsml_filenames.append(f'{subseqname}')

        ##########
        # STEP 3:
        # Make the main .sml file

        asset_names: List[List[str]] = []
        seqings = []
        subseq_positions: List[int] = []
        for pos1 in seq.keys():
            pos_seqs = seq[pos1]['sequencing']

            pos_seqs['twait'] = pos_seqs.get('twait', 0)
            pos_seqs['nrep'] = pos_seqs.get('nrep', 1)
            pos_seqs['jump_input'] = pos_seqs.get('jump_input', 0)
            pos_seqs['jump_target'] = pos_seqs.get('jump_target', 0)
            pos_seqs['goto'] = pos_seqs.get('goto', 0)
            seqings.append(pos_seqs)
            if seq[pos1]['type'] == 'subsequence':
                subseq_positions.append(pos1)
                asset_names.append([sn for sn in subseqsml_filenames
                                    if f'_{pos1}' in sn])
            else:
                asset_names.append([wn for wn in wfmx_filenames
                                    if f'wfm_{pos1}' in wn])
        seqing = {k: [d[k] for d in seqings] for k in seqings[0].keys()}

        log.debug(f'Assets for SML file: {asset_names}')

        mainseqname = seqname
        mainseqsml = BaselAWG5204._makeSMLFile(trig_waits=seqing['twait'],
                                            nreps=seqing['nrep'],
                                            event_jumps=seqing['jump_input'],
                                            event_jump_to=seqing['jump_target'],
                                            go_to=seqing['goto'],
                                            elem_names=asset_names,
                                            seqname=mainseqname,
                                            chans=len(channel_mapping),
                                            subseq_positions=subseq_positions)

        ##########
        # STEP 4:
        # Build the .seqx file

        user_file = b''
        setup_file = BaselAWG5204._makeSetupFile(mainseqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode='a')
        for ssn, ssf in zip(subseqsml_filenames, subseqsml_files):
            zipfile.writestr(f'Sequences/{ssn}.sml', ssf)
        zipfile.writestr(f'Sequences/{mainseqname}.sml', mainseqsml)

        for (name, wfile) in zip(wfmx_filenames, wfmx_files):
            zipfile.writestr(f'Waveforms/{name}.wfmx', wfile)

        zipfile.writestr('setup.xml', setup_file)
        zipfile.writestr('userNotes.txt', user_file)
        zipfile.close()

        buffer.seek(0)
        seqx = buffer.getvalue()
        buffer.close()

        return seqx
    
        
    #copied this static method because we want to call the new version of _makeSMLFile
    @staticmethod
    def makeSEQXFile(trig_waits: Sequence[int],
                     nreps: Sequence[int],
                     event_jumps: Sequence[int],
                     event_jump_to: Sequence[int],
                     go_to: Sequence[int],
                     wfms: Sequence[Sequence[np.ndarray]],
                     amplitudes: Sequence[float],
                     seqname: str,
                     flags: Optional[Sequence[Sequence[Sequence[int]]]] = None
                     ) -> bytes:
        """
        Make a full .seqx file (bundle)
        A .seqx file can presumably hold several sequences, but for now
        we support only packing a single sequence
        For a single sequence, a .seqx file is a bundle of two files and
        two folders:
        /Sequences
            sequence.sml
        /Waveforms
            wfm1.wfmx
            wfm2.wfmx
            ...
        setup.xml
        userNotes.txt
        Args:
            trig_waits: Wait for a trigger? If yes, you must specify the
                trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            nreps: No. of repetitions. 0 corresponds to infinite.
            event_jumps: Jump when event triggered? If yes, you must specify
                the trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            event_jump_to: Jump target in case of event. 1-indexed,
                0 means next. Must be specified for all elements.
            go_to: Which element to play next. 1-indexed, 0 means next.
            wfms: numpy arrays describing each waveform plus two markers,
                packed like np.array([wfm, m1, m2]). These numpy arrays
                are then again packed in lists according to:
                [[wfmch1pos1, wfmch1pos2, ...], [wfmch2pos1, ...], ...]
            amplitudes: The peak-to-peak amplitude in V of the channels, i.e.
                a list [ch1_amp, ch2_amp].
            seqname: The name of the sequence. This name will appear in the
                sequence list. Note that all spaces are converted to '_'
            flags: Flags for the auxiliary outputs. 0 for 'No change', 1 for
                'High', 2 for 'Low', 3 for 'Toggle', or 4 for 'Pulse'. 4 flags
                [A, B, C, D] for every channel in every element, packed like:
                [[ch1pos1, ch1pos2, ...], [ch2pos1, ...], ...]
                If omitted, no flags will be set.
        Returns:
            The binary .seqx file, ready to be sent to the instrument.
        """

        # input sanitising to avoid spaces in filenames
        seqname = seqname.replace(' ', '_')

        (chans, elms) = (len(wfms), len(wfms[0]))
        wfm_names = [[f'wfmch{ch}pos{el}' for ch in range(1, chans+1)]
                     for el in range(1, elms+1)]

        # generate wfmx files for the waveforms
        flat_wfmxs = [] # type: List[bytes]
        for amplitude, wfm_lst in zip(amplitudes, wfms):
            flat_wfmxs += [AWG70000A.makeWFMXFile(wfm, amplitude)
                           for wfm in wfm_lst]

        # This unfortunately assumes no subsequences
        flat_wfm_names = list(np.reshape(np.array(wfm_names).transpose(),
                                         (chans*elms,)))

        sml_file = BaselAWG5204._makeSMLFile(trig_waits, nreps,
                                          event_jumps, event_jump_to,
                                          go_to, wfm_names,
                                          seqname,
                                          chans, flags=flags)

        user_file = b''
        setup_file = BaselAWG5204._makeSetupFile(seqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode='a')
        zipfile.writestr(f'Sequences/{seqname}.sml', sml_file)

        for (name, wfile) in zip(flat_wfm_names, flat_wfmxs):
            zipfile.writestr(f'Waveforms/{name}.wfmx', wfile)

        zipfile.writestr('setup.xml', setup_file)
        zipfile.writestr('userNotes.txt', user_file)
        zipfile.close()

        buffer.seek(0)
        seqx = buffer.getvalue()
        buffer.close()

        return seqx

    
    #overwrite this method to allow for control over jump timing   
    #NOTE: this only fixes the problem when NOT using forged sequences
    @staticmethod
    def _makeSMLFile(trig_waits: Sequence[int],
                     nreps: Sequence[int],
                     event_jumps: Sequence[int],
                     event_jump_to: Sequence[int],
                     go_to: Sequence[int],
                     elem_names: Sequence[Sequence[str]],
                     seqname: str,
                     chans: int,
                     subseq_positions: Sequence[int] = (),
                     flags: Optional[Sequence[Sequence[Sequence[int]]]] = None
                     ) -> str:
        """
        Make an xml file describing a sequence.
        Args:
            trig_waits: Wait for a trigger? If yes, you must specify the
                trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            nreps: No. of repetitions. 0 corresponds to infinite.
            event_jumps: Jump when event triggered? If yes, you must specify
                the trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            event_jump_to: Jump target in case of event. 1-indexed,
                0 means next. Must be specified for all elements.
            go_to: Which element to play next. 1-indexed, 0 means next.
            elem_names: The waveforms/subsequences to use. Should be packed
                like:
                [[wfmpos1ch1, wfmpos1ch2, ...],
                 [subseqpos2],
                 [wfmpos3ch1, wfmpos3ch2, ...], ...]
            seqname: The name of the sequence. This name will appear in
                the sequence list of the instrument.
            chans: The number of channels. Can not be inferred in the case
                of a sequence containing only subsequences, so must be provided
                up front.
            subseq_positions: The positions (step numbers) occupied by
                subsequences
            flags: Flags for the auxiliary outputs. 0 for 'No change', 1 for
                'High', 2 for 'Low', 3 for 'Toggle', or 4 for 'Pulse'. 4 flags
                [A, B, C, D] for every channel in every element, packed like:
                [[ch1pos1, ch1pos2, ...], [ch2pos1, ...], ...]
                If omitted, no flags will be set.
        Returns:
            A str containing the file contents, to be saved as an .sml file
        """

        offsetdigits = 9

        waitinputs = {0: 'None', 1: 'TrigA', 2: 'TrigB', 3: 'Internal'}
        eventinputs = {0: 'None', 1: 'TrigA', 2: 'TrigB', 3: 'Internal'}
        flaginputs = {0:'NoChange', 1:'High', 2:'Low', 3:'Toggle', 4:'Pulse'}

        inputlsts = [trig_waits, nreps, event_jump_to, go_to]
        lstlens = [len(lst) for lst in inputlsts]
        if lstlens.count(lstlens[0]) != len(lstlens):
            raise ValueError('All input lists must have the same length!')

        if lstlens[0] == 0:
            raise ValueError('Received empty sequence option lengths!')

        if lstlens[0] != len(elem_names):
            raise ValueError('Mismatch between number of waveforms and'
                             ' number of sequencing steps.')

        N = lstlens[0]

        # form the timestamp string
        timezone = time.timezone
        tz_m, _ = divmod(timezone, 60)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = '-'
            tz_h *= -1
        else:
            signstr = '+'
        timestr = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        timestr += signstr
        timestr += f'{tz_h:02.0f}:{tz_m:02.0f}'

        datafile = ET.Element('DataFile', attrib={'offset': '0'*offsetdigits,
                                                  'version': '0.1'})
        dsc = ET.SubElement(datafile, 'DataSetsCollection')
        dsc.set("xmlns", "http://www.tektronix.com")
        dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        dsc.set("xsi:schemaLocation", (r"http://www.tektronix.com file:///" +
                                       r"C:\Program%20Files\Tektronix\AWG70000" +
                                       r"\AWG\Schemas\awgSeqDataSets.xsd"))
        datasets = ET.SubElement(dsc, 'DataSets')
        datasets.set('version', '1')
        datasets.set("xmlns", "http://www.tektronix.com")

        # Description of the data
        datadesc = ET.SubElement(datasets, 'DataDescription')
        temp_elem = ET.SubElement(datadesc, 'SequenceName')
        temp_elem.text = seqname
        temp_elem = ET.SubElement(datadesc, 'Timestamp')
        temp_elem.text = timestr
        temp_elem = ET.SubElement(datadesc, 'JumpTiming')
        temp_elem.text = BaselAWG5204.JUMP_TIMING  # TODO: What does this control? THIS CONTROLS THE JUMP TIMING!
        temp_elem = ET.SubElement(datadesc, 'RecSampleRate')
        temp_elem.text = 'NaN'
        temp_elem = ET.SubElement(datadesc, 'RepeatFlag')
        temp_elem.text = 'false'
        temp_elem = ET.SubElement(datadesc, 'PatternJumpTable')
        temp_elem.set('Enabled', 'false')
        temp_elem.set('Count', '65536')
        steps = ET.SubElement(datadesc, 'Steps')
        steps.set('StepCount', f'{N:d}')
        steps.set('TrackCount', f'{chans:d}')

        for n in range(1, N+1):
            step = ET.SubElement(steps, 'Step')
            temp_elem = ET.SubElement(step, 'StepNumber')
            temp_elem.text = f'{n:d}'
            # repetitions
            rep = ET.SubElement(step, 'Repeat')
            repcount = ET.SubElement(step, 'RepeatCount')
            if nreps[n-1] == 0:
                rep.text = 'Infinite'
                repcount.text = '1'
            elif nreps[n-1] == 1:
                rep.text = 'Once'
                repcount.text = '1'
            else:
                rep.text = "RepeatCount"
                repcount.text = f"{nreps[n-1]:d}"
            # trigger wait
            temp_elem = ET.SubElement(step, 'WaitInput')
            temp_elem.text = waitinputs[trig_waits[n-1]]
            # event jump
            temp_elem = ET.SubElement(step, 'EventJumpInput')
            temp_elem.text = eventinputs[event_jumps[n-1]]
            jumpto = ET.SubElement(step, 'EventJumpTo')
            jumpstep = ET.SubElement(step, 'EventJumpToStep')
            if event_jump_to[n-1] == 0:
                jumpto.text = 'Next'
                jumpstep.text = '1'
            else:
                jumpto.text = "StepIndex"
                jumpstep.text = f"{event_jump_to[n-1]:d}"
            # Go to
            goto = ET.SubElement(step, 'GoTo')
            gotostep = ET.SubElement(step, 'GoToStep')
            if go_to[n-1] == 0:
                goto.text = 'Next'
                gotostep.text = '1'
            else:
                goto.text = "StepIndex"
                gotostep.text = f"{go_to[n-1]:d}"

            assets = ET.SubElement(step, 'Assets')
            for assetname in elem_names[n-1]:
                asset = ET.SubElement(assets, 'Asset')
                temp_elem = ET.SubElement(asset, 'AssetName')
                temp_elem.text = assetname
                temp_elem = ET.SubElement(asset, 'AssetType')
                if n in subseq_positions:
                    temp_elem.text = 'Sequence'
                else:
                    temp_elem.text = 'Waveform'

            # convert flag settings to strings
            flags_list = ET.SubElement(step, 'Flags')
            for chan in range(chans):
                flagset = ET.SubElement(flags_list, 'FlagSet')
                for flgind, flg in enumerate(['A', 'B', 'C', 'D']):
                    temp_elem = ET.SubElement(flagset, 'Flag')
                    temp_elem.set('name', flg)
                    if flags is None:
                        # no flags were passed to the function
                        temp_elem.text = 'NoChange'
                    else:
                        temp_elem.text = flaginputs[flags[chan][n-1][flgind]]

        temp_elem = ET.SubElement(datasets, 'ProductSpecific')
        temp_elem.set('name', '')
        temp_elem = ET.SubElement(datafile, 'Setup')

        # the tostring() call takes roughly 75% of the total
        # time spent in this function. Can we speed up things?
        # perhaps we should use lxml?
        xmlstr = ET.tostring(datafile, encoding='unicode')
        xmlstr = xmlstr.replace('><', '>\r\n<')

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace('0'*offsetdigits,
                                '{num:0{pad}d}'.format(num=len(xmlstr),
                                                       pad=offsetdigits))

        return xmlstr