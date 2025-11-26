import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, sample_rate=16000, trim_silence=True, trim_cfg=None):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.trim_silence = trim_silence
        self.sr = sample_rate
        
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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        if self.trim_silence:
            X_trim = trim_silence_hybrid(
                X,
                sample_rate=self.sr,
                frame_size_ms=self.trim_cfg['frame_size_ms'],
                energy_threshold=self.trim_cfg['energy_threshold'],
                zcr_threshold=self.trim_cfg['zcr_threshold'],
                hangover_ms=self.trim_cfg['hangover_ms'],
                smooth_win=self.trim_cfg['smooth_win']
            )
            # if trimming return empty, keep original
            if X_trim.size > 0:
                X = X_trim
        
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, sample_rate=16000, trim_silence=True, trim_cfg=None):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.trim_silence = trim_silence
        self.sr = sample_rate
        
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

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        if self.trim_silence:
            X_trim = trim_silence_hybrid(
                X,
                sample_rate=self.sr,
                frame_size_ms=self.trim_cfg['frame_size_ms'],
                energy_threshold=self.trim_cfg['energy_threshold'],
                zcr_threshold=self.trim_cfg['zcr_threshold'],
                hangover_ms=self.trim_cfg['hangover_ms'],
                smooth_win=self.trim_cfg['smooth_win']
            )
            # if trimming return empty, keep original
            if X_trim.size > 0:
                X = X_trim
        
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    

def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


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