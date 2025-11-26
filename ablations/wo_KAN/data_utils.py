import os
import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import Dataset

from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav

"""
Create dataset for ASVspoof2019-attr17:
- flac files
- grouth truth
Create ground truth for 2 modules:
- Multi-task learning model:
    + Ground-truth includes attributes for each task (AS1, AS2, etc.)
- KAN model:
    + Ground-truth is the attack type of sample (A01, A02, etc.)
"""


class Dataset_ASVspoof2019_attr17(Dataset):
    def __init__(
        self, 
        flac_files, 
        ground_truths, 
        base_dir,
        phase='train',
        sample_rate=16000,
        cut=64600, 
        trim_silence=True,
        trim_cfg=None,
        rawboost_args=None,
        rawboost_mix_probs=None,
        ):
        """
        Initialize the dataset with embeddings and ground truth labels.
        Args:
            flac_files (list): list of flac audio files
            ground_truth (dict): Dictionary containing ground truth labels for each task.
        """
        self.flac_files = flac_files
        self.ground_truths = ground_truths
        self.base_dir = base_dir
        self.phase = phase
        self.sr = sample_rate
        self.cut = cut
        self.trim_silence = trim_silence
        self.rawboost_args = rawboost_args
        
        # default config for silence trimming
        default_trim_cfg = dict(
            sample_rate=16000,
            frame_size_ms=20,
            energy_threshold=0.01,
            zcr_threshold=0.08,
            hangover_ms=200,
            smooth_win=3,
        )
        if trim_cfg is not None:
            default_trim_cfg.update(trim_cfg)
        self.trim_cfg = default_trim_cfg
        
        # default mixing probs for RawBoost
        if rawboost_mix_probs is None:
            rawboost_mix_probs = {
                0: 0.04,  # none
                1: 0.12,  # LnL
                2: 0.12,  # ISD
                3: 0.12,  # SSI
                4: 0.12,  # 1+2+3
                5: 0.12,  # 1+2
                6: 0.12,  # 1+3
                7: 0.12,  # 2+3
                8: 0.12   # 1||2
            }
        # normalize to sum to 1
        total = sum(rawboost_mix_probs.values())
        self.rawboost_mix_probs = {k: v/total for k, v in rawboost_mix_probs.items()}
        self.rb_combos, self.rb_weights = zip(*self.rawboost_mix_probs.items())
        self.rb_weights = np.array(self.rb_weights, dtype=np.float64)
        self.rb_weights = self.rb_weights / self.rb_weights.sum()

    def __len__(self):
        return len(self.flac_files)
    
    def _sample_rb_combo(self) -> int:
        # choose the random RawBoost algo
        return int(np.random.choice(self.rb_combos, p=self.rb_weights))
    
    def __getitem__(self, index):
        """
        Get the item at the specified index.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (flac_file, ground_truth) for the specified index.
        """

        key = self.flac_files[index]
        
        # 1) Load
        x, _ = librosa.load(os.path.join(self.base_dir, f"flac/{key}.flac"), sr=self.sr)
        
        # 2) Data pre-processing: Silence trimming
        if self.trim_silence:
            x_trim = trim_silence_hybrid(
                x,
                sample_rate=self.sr,
                frame_size_ms=self.trim_cfg['frame_size_ms'],
                energy_threshold=self.trim_cfg['energy_threshold'],
                zcr_threshold=self.trim_cfg['zcr_threshold'],
                hangover_ms=self.trim_cfg['hangover_ms'],
                smooth_win=self.trim_cfg['smooth_win']
            )
            # if trimming return empty, keep original
            if x_trim.size > 0:
                x = x_trim
        
        # 3) Data augmentation: RawBoost
        # only for training phase
        if self.phase == 'train':
            assert self.rawboost_args is not None, "RawBoost args must be provided for training phase"
            combo = self._sample_rb_combo()
            x = process_Rawboost_feature(x, self.sr, self.rawboost_args, combo)
        
        # 4) Fixed length (train: random pad; dev and eval: pad)
        if self.phase == 'train':
            x_pad = pad_random(x, self.cut)
        elif self.phase in ['dev', 'eval']:
            x_pad = pad(x, self.cut)
        
        # 5) To tensor + label
        x_inp = torch.Tensor(x_pad)

        attribute_labels = {
            task: torch.tensor(self.ground_truths[index][task], dtype=torch.long).squeeze(0)
                for task in self.ground_truths[index]
        }

        return x_inp, attribute_labels


ASVSpoof2019_Attr17_attack_attribute_structure = [
    ([0], [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]),
    ([1], [4, 5, 15, 16]),
    ([2], [12, 13, 14]),
    ([3], [0, 1, 2, 3, 6, 7, 8, 11]),
    ([4], [4, 12, 15]),
    ([5], [5]),
    ([6], [9, 10]),
    ([7], [13, 14]),
    ([8], [16]),
    ([9], [0, 1, 7]),
    ([10], [2]),
    ([11], [6, 8, 11]),
    ([12], [9, 10]),
    ([13], [12]),
    ([14], [3, 4, 5, 13, 14, 15, 16]),
    ([15], [0, 1, 7]),
    ([16], [2]),
    ([17], [3]),
    ([18], [4, 15]),
    ([19], [5]),
    ([20], [6, 8, 11, 13, 14]),
    ([21], [9, 10]),
    ([22], [12]),
    ([23], [16]),
    ([24], [0, 1]),
    ([25], [2, 4, 6, 7, 8, 11, 15]),
    ([26], [9, 10]),
    ([27], [16]),
    ([28], [3, 5, 12, 13, 14]),
    ([29], [0, 7, 8, 14, 15]),
    ([30], [1, 2, 13]),
    ([31], [3]),
    ([32], [4]),
    ([33], [5]),
    ([34], [6]),
    ([35], [9, 10]),
    ([36], [11]),
    ([37], [12]),
    ([38], [16]),
    ([39], [0, 11, 14]),
    ([40], [1, 2, 4, 6]),
    ([41], [3]),
    ([42], [5]),
    ([43], [7]),
    ([44], [8]),
    ([45], [9]),
    ([46], [10]),
    ([47], [12, 15]),
    ([48], [13]),
    ([49], [16])
]


def get_labels(source):

    # data based on "Table 1: Summary of LA spoofing systems." in "ASVspoof 2019: a large-scale public database of synthetized, converted and replayed speech"
    
    # the data is not balanced when it comes to the attributes (e.g. Text(input) has more samples than MCC-F0(output))
    data = {
            "A01":      [[0], [0], [0], [0], [0], [0], [0], [0]],
            "A02":      [[1], [0], [0], [0], [0], [0], [1], [1]],
            "A03":      [[2], [0], [0], [1], [1], [1], [1], [1]],
            "A04-16":   [[3], [0], [0], [5], [2], [4], [2], [2]],
            "A05":      [[4], [1], [1], [5], [3], [1], [3], [1]],
            "A06-19":   [[5], [1], [2], [5], [4], [4], [4], [3]],

            "A07":      [[6], [0], [0], [2], [5], [1], [5], [1]],
            "A08":      [[7], [0], [0], [0], [0], [1], [0], [4]],
            "A09":      [[8], [0], [0], [2], [5], [1], [0], [5]],
            "A10":      [[9], [0], [3], [3], [6], [2], [6], [6]], 
            "A11":      [[10], [0], [3], [3], [6], [2], [6], [7]],
            "A12":      [[11], [0], [0], [2], [5], [1], [7], [0]],
            "A13":      [[12], [2], [1], [4], [7], [4], [8], [8]],
            "A14":      [[13], [2], [4], [5], [5], [4], [1], [9]],
            "A15":      [[14], [2], [4], [5], [5], [4], [0], [0]],
            "A17":      [[15], [1], [1], [5], [3], [1], [0], [8]],
            "A18":      [[16], [1], [5], [5], [8], [3], [9], [10]],
        } 
    
    attribute_sets = ["AS1", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7"]
    
    labels = {}
    for attack_type, ys in data.items():
        labels[attack_type] = {
            "Attack_id": ys[0],
            **{f'T{i+1}_{attribute_sets[i]}': ys[i+1] for i in range(len(attribute_sets))},
        }
    
    return labels[source]


def load_protocol_data(protocols_path):
    """
    Load protocol data from the specified path.
    Args:
        protocols_path (str): Path to the protocols file.
    Returns:
        pd.DataFrame: Loaded protocol data as a pandas DataFrame.
    """
    if not os.path.exists(protocols_path):
        raise FileNotFoundError(f"Protocols file not found at {protocols_path}")
    df = pd.read_csv(protocols_path, sep=" ", header=None)
    
    return df


def fetch_protocol(protocols_path):
    """
    Create list of flac files and ground truth for multi-task learning based on protocols.
    Args:
        protocols_path (str): Path to the protocols file.
    """
    protocols_data = load_protocol_data(protocols_path)
    
    flac_files = []
    ground_truths = []
    
    for index, row in protocols_data.iterrows():
        
        flac_id = row[1]
        source = row[3]

        flac_files.append(flac_id)

        attributes = get_labels(source)
        ground_truths.append(attributes)

    flac_files = np.array(flac_files)    
    ground_truths = np.array(ground_truths, dtype=object)
    return flac_files, ground_truths


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


### ===========================================================================
### Data pre-processing: Silence trimming using STE + ZCR
### ===========================================================================
def trim_silence_hybrid(
        wav,
        sample_rate=16000,
        frame_size_ms=20,
        energy_threshold=0.01,
        zcr_threshold=0.08,
        hangover_ms=200,
        smooth_win=3
    ):
    """
    Trims leading/trailing silence using Short-Time Energy (STE) + Zero Crossing Rate (ZCR).

    Args:
        wav (np.ndarray): 1D audio waveform.
        sample_rate (int): Sampling rate.
        frame_size_ms (int): Frame size in ms (non-overlapping frames).
        energy_threshold (float): STE threshold (after max-norm) to declare speech.
        zcr_threshold (float): ZCR per-sample crossing rate (0..1) to help keep unvoiced speech.
        hangover_ms (int): Extra padding kept on both sides of detected speech (ms).
        smooth_win (int): Moving window (in frames) to smooth the speech mask (odd number recommended).

    Returns:
        np.ndarray: Trimmed waveform (possibly empty if no speech is found).
    """
    if wav.size == 0:
        return wav.astype(np.float32)

    frame_size = int(sample_rate * frame_size_ms / 1000)
    if frame_size <= 0:
        frame_size = 1

    num_frames = int(np.ceil(len(wav) / frame_size))

    # ------ Short-Time Energy (STE) ------
    energies = np.array([
        np.sum(wav[i*frame_size : min((i+1)*frame_size, len(wav))]**2)
        for i in range(num_frames)
    ], dtype=np.float64)

    # Normalize energies to [0, 1] (robust to absolute level)
    max_e = np.max(energies) + 1e-12
    energies = energies / max_e

    # ------ Zero Crossing Rate (ZCR) ------
    # ZCR per frame as fraction of sign changes per sample (0..1)
    zcrs = []
    for i in range(num_frames):
        seg = wav[i*frame_size : min((i+1)*frame_size, len(wav))]
        if seg.size < 2:
            zcrs.append(0.0)
            continue
        # Count sign changes; treat zeros by shifting sign with epsilon
        s = np.sign(seg + 1e-12)
        zc = np.mean(s[:-1] * s[1:] < 0)  # fraction of sign flips
        zcrs.append(zc)
    zcrs = np.asarray(zcrs, dtype=np.float64)

    # ------ Combine STE + ZCR ------
    # Primary gate: energy above threshold
    # Secondary gate: if energy is a bit low but ZCR is high (unvoiced consonants),
    # keep it if energy > 30% of energy threshold and ZCR > zcr_threshold.
    energy_gate = energies > energy_threshold
    rescue_gate = (energies > (0.3 * energy_threshold)) & (zcrs > zcr_threshold)
    speech_mask = energy_gate | rescue_gate

    # ------ Smooth the mask to avoid choppy cuts ------
    if smooth_win and smooth_win > 1:
        k = np.ones(int(smooth_win), dtype=np.float64)
        smoothed = np.convolve(speech_mask.astype(np.float64), k, mode='same') / np.sum(k)
        speech_mask = smoothed > 0.5

    # ------ Hangover (pad around detected speech) ------
    hangover_frames = max(1, int(np.round((hangover_ms / 1000.0) * sample_rate / frame_size)))
    if hangover_frames > 0 and speech_mask.any():
        k = np.ones(2 * hangover_frames + 1, dtype=np.float64)
        padded = np.convolve(speech_mask.astype(np.float64), k, mode='same') > 0
        speech_mask = padded

    voiced_indices = np.where(speech_mask)[0]
    if voiced_indices.size == 0:
        return np.array([], dtype=np.float32)

    start_frame = voiced_indices[0]
    end_frame = voiced_indices[-1]

    start_sample = max(0, start_frame * frame_size)
    end_sample = min((end_frame + 1) * frame_size, len(wav))

    return wav[start_sample:end_sample].astype(np.float32)


### ===========================================================================
### Data augmentation: RawBoost
### ===========================================================================
def process_Rawboost_feature(feature, sr, args, algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        feature=feature
    
    return feature


### ===========================================================================
### Create dataset for ASVspoof2019-attr2 evalset - bonafide only
### ===========================================================================

class Dataset_ASVspoof2019_attr2_flacOnly(Dataset):
    def __init__(self, flac_files, base_dir, ST=False):
        
        self.flac_files = flac_files
        self.base_dir = base_dir
        self.cut = 64000
        self.ST = ST
        
    def __len__(self):
        return len(self.flac_files)
    
    def __getitem__(self, index):

        key = self.flac_files[index]
        X, _ = sf.read(os.path.join(self.base_dir, f"flac/{key}.flac"))

        if not self.ST:
            X_pad = pad(X, self.cut)
        else:
            X_pad = X
        
        x_inp = torch.Tensor(X_pad)

        return x_inp, _
    

def fetch_asvspoof2019_attr2_phase_bonafide(protocols_path):
    
    protocols_data = load_protocol_data(protocols_path)
    
    flac_files = []
    
    for index, row in protocols_data.iterrows():
        
        flac_id = row[1]
        label = row[4]
        
        if label == 'spoof':
            continue

        flac_files.append(flac_id)

    flac_files = np.array(flac_files) 
       
    return flac_files


def fetch_asvspoof2019_attr2_phase_full(protocols_path):
    
    protocols_data = load_protocol_data(protocols_path)
    
    flac_files = []
    labels = []
    
    for index, row in protocols_data.iterrows():
        
        flac_id = row[1]
        label = 1 if row[4] == 'bonafide' else 0
        
        flac_files.append(flac_id)
        labels.append(label)

    flac_files = np.array(flac_files)   
    labels = np.array(labels, dtype=np.int64)
     
    return flac_files, labels 


def collate_fn_bonafide(batch):
    """
    args:
    =====
        - batch: list of tuples (waveform, label) for train
        
    returns:
    ========
        - (batch_x, labels_tensor)
    """
    max_len = 64600  
    sample_rate = 16000

    batch_x = []
    gt_list = []
    
    for waveform, gt in batch:
        
        # convert to numpy
        if isinstance(waveform, torch.Tensor):
            wav = waveform.squeeze().cpu().numpy()
        else:
            wav = waveform

        # trim silence
        trimmed = trim_audio_silence(wav, sample_rate=sample_rate)
        if trimmed.size == 0:
            trimmed = wav

        # pad to max_len 
        padded_np = pad(trimmed, max_len=max_len)

        # convert to tensor [1, T]
        padded_tensor = torch.from_numpy(padded_np).float().unsqueeze(0)

        batch_x.append(padded_tensor)

        # collect label
        gt_list.append(gt)

    # stack
    batch_x = torch.stack(batch_x, dim=0)  # (#bs, 1, T)
    batch_x = batch_x.squeeze(1)  # (#bs, T)

    gt = torch.tensor(gt_list)

    return batch_x, gt
