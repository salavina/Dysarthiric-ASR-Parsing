## helper classes ##
__author__ = 'Saeid Alavi Naeini'

# required imports
import tensorflow
import os
import librosa
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from numpy import linalg as LA

import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.spatial import distance
from matplotlib import rcParamsDefault
from pydub import AudioSegment

import pandas as pd
import json
import matplotlib.patches as mpatches


# add required json files
# Create a dictunary, repetition #: audio file path
reference_audio_dict = {
    1: "./reference_audio/1_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    2: "./reference_audio/2_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    3: "./reference_audio/3_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    4: "./reference_audio/4_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    5: "./reference_audio/5_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    6: "./reference_audio/6_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    7: "./reference_audio/7_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    8: "./reference_audio/8_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    9: "./reference_audio/9_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    10: "./reference_audio/10_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    11: "./reference_audio/11_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    12: "./reference_audio/12_NF01_02_SBK_20170515_BBP_NORMAL_al.wav",
    13: "./reference_audio/13_NF01_02_SBK_20170515_BBP_NORMAL_al.wav"
}

# Create a dictunary, repetition #: parsed file path
reference_parse_dict_whole = {
    1: "./reference_parse/1_NF01_02_BBP_NORMAL_audio.Table",
    2: "./reference_parse/2_NF01_02_BBP_NORMAL_audio.Table",
    3: "./reference_parse/3_NF01_02_BBP_NORMAL_audio.Table",
    4: "./reference_parse/4_NF01_02_BBP_NORMAL_audio.Table",
    5: "./reference_parse/5_NF01_02_BBP_NORMAL_audio.Table",
    6: "./reference_parse/6_NF01_02_BBP_NORMAL_audio.Table",
    7: "./reference_parse/7_NF01_02_BBP_NORMAL_audio.Table",
    8: "./reference_parse/8_NF01_02_BBP_NORMAL_audio.Table",
    9: "./reference_parse/9_NF01_02_BBP_NORMAL_audio.Table",
    10: "./reference_parse/10_NF01_02_BBP_NORMAL_audio.Table",
    11: "./reference_parse/11_NF01_02_BBP_NORMAL_audio.Table",
    12: "./reference_parse/12_NF01_02_BBP_NORMAL_audio.Table",
    13: "./reference_parse/13_NF01_02_BBP_NORMAL_audio.Table"
}

# Create a dictunary, repetition #: parsed file path
reference_parse_dict = {
    1: "./reference_parse_word_precision/1_NF01_02_BBP_NORMAL_audio.Table",
    2: "./reference_parse_word_precision/2_NF01_02_BBP_NORMAL_audio.Table",
    3: "./reference_parse_word_precision/3_NF01_02_BBP_NORMAL_audio.Table",
    4: "./reference_parse_word_precision/4_NF01_02_BBP_NORMAL_audio.Table",
    5: "./reference_parse_word_precision/5_NF01_02_BBP_NORMAL_audio.Table",
    6: "./reference_parse_word_precision/6_NF01_02_BBP_NORMAL_audio.Table",
    7: "./reference_parse_word_precision/7_NF01_02_BBP_NORMAL_audio.Table",
    8: "./reference_parse_word_precision/8_NF01_02_BBP_NORMAL_audio.Table",
    9: "./reference_parse_word_precision/9_NF01_02_BBP_NORMAL_audio.Table",
    10: "./reference_parse_word_precision/10_NF01_02_BBP_NORMAL_audio.Table",
    11: "./reference_parse_word_precision/11_NF01_02_BBP_NORMAL_audio.Table",
    12: "./reference_parse_word_precision/12_NF01_02_BBP_NORMAL_audio.Table",
    13: "./reference_parse_word_precision/13_NF01_02_BBP_NORMAL_audio.Table"
}

class ASRLoader:
    def __init__(self, PATH_TO_VOCAB_FILE, PATH_TO_WEIGHTS_FOLDER):
        self.PATH_TO_VOCAB_FILE = PATH_TO_VOCAB_FILE
        self.PATH_TO_WEIGHTS_FOLDER = PATH_TO_WEIGHTS_FOLDER

    # function that loads the ML model
    def load_model(self):
        tokenizer = Wav2Vec2CTCTokenizer(self.PATH_TO_VOCAB_FILE, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained(self.PATH_TO_WEIGHTS_FOLDER).cuda()
        return model, processor, feature_extractor

    # apply ML model to transcribe audio
    # you would just use the audio recorded in front-end as the input
    def apply_ml(self, file_Sample, sr):
        model, processor, feature_extractor = self.load_model()
        # file_Sample, sr = librosa.load(SPEECH_FILE)
        file_Sample = librosa.resample(file_Sample, orig_sr=sr, target_sr=16000)
        inputs = feature_extractor(file_Sample, sampling_rate=16000, return_tensors="pt")
        inputs.to('cuda')
        with torch.no_grad():
          logits = model(**inputs).logits
        predicted_class_ids = torch.argmax(logits, dim=-1)
        predicted_asr = processor.batch_decode(predicted_class_ids)[0]
        # each repetition has to have one of 3 words in it, so we count all 3 words occurences and take the max as # of repetitions
        buy_oc = predicted_asr.count('buy')
        bobby_oc = predicted_asr.count('bobby')
        puppy_oc = predicted_asr.count('puppy')
        counts_num_min = min(buy_oc, bobby_oc, puppy_oc)
        counts_num_max = max(buy_oc, bobby_oc, puppy_oc)
        # counts_num_max
        transcription = processor.batch_decode(predicted_class_ids)[0]
        return counts_num_max, transcription


### IoU Calculator
class AudioParser:
    """Audio Parser class that utilizes DTW to parse new signals based on parsed reference file.
      Parameters
      ----------
      x: Array
        output audio array from librosa.load (reference)
      fs1: int
        sampling frequency output from librosa.load (reference)
      fs2: int
        sampling frequency output from librosa.load (audio file to be parsed)
      start_lst: list
          reference parsings for start of each repetition
      end_lst: list
          reference parsings for end of each repetition
      path: list of tuples
          DTW warping path
      path: int - default = 512
          hop size of MFCC
      """

    def __init__(self, fs1, fs2, reference_txt_path, target_txt_path=None, path=None, hop_size=512):
        self.reference_text_path = reference_txt_path
        self.target_txt_path = target_txt_path
        self.path = path
        self.hop_size = hop_size
        self.fs1 = fs1
        self.fs2 = fs2
        # setting threshold
        self.THRESH = 0.75

    def time_to_position_converter(self, time):
        # total duration of audio file
        # dur = librosa.get_duration(self.x, sr=self.fs1)
        # get position of time in audio array
        position_audio = round(time * self.fs1)
        # map position into mfcc array
        position_mfcc = round(position_audio / self.hop_size)
        return position_mfcc

    def position_to_time_converter(self, position_mfcc):
        position_audio = position_mfcc * self.hop_size
        time = position_audio / self.fs2
        return time

    def start_end_list_generator(self):
        start_lst = []
        end_lst = []
        f = open(self.reference_text_path, 'r')
        for line in f:
            if ('rep' in line.lower() or 'bbp' in line.lower()):
                a = line.split(',')
                start_lst.append(float(a[0]))
                end_lst.append(float(a[2]))
        return start_lst, end_lst

    def start_end_list_generator_target(self):
        start_lst = []
        end_lst = []
        start_lst_new = []
        end_lst_new = []
        with open(self.target_txt_path, 'r') as f:
            data = json.load(f)
        start_lst = data['BBP_parsed'][0]
        end_lst = data['BBP_parsed'][1]
        for i in range(0, len(start_lst), 4):
            start_lst_new.append(start_lst[i])
        for i in range(3, len(end_lst), 4):
            end_lst_new.append(end_lst[i])
        return start_lst_new, end_lst_new

    def audio_parser(self):
        start_lst_parsed_temp = []
        end_lst_parsed_temp = []
        start_lst_parsed = []
        end_lst_parsed = []
        start_lst, end_lst = self.start_end_list_generator()
        for i, j in zip(start_lst, end_lst):
            t_start = self.time_to_position_converter(i)
            t_end = self.time_to_position_converter(j)
            for k in self.path:
                if k[0] == t_start:
                    start_lst_parsed_temp.append(self.position_to_time_converter(k[1]))
                elif k[0] == t_end:
                    end_lst_parsed_temp.append(self.position_to_time_converter(k[1]))
            if start_lst_parsed_temp:
                start_lst_parsed.append(start_lst_parsed_temp[len(start_lst_parsed_temp) - 1])
            if end_lst_parsed_temp:
                end_lst_parsed.append(end_lst_parsed_temp[len(end_lst_parsed_temp) - 1])
            start_lst_parsed_temp = []
            end_lst_parsed_temp = []
        return start_lst_parsed, end_lst_parsed

    def bbp_speech_duration_generator_target(self):
        lst_dur_BBP = []
        start_lst, end_lst = self.audio_parser()
        for i in range(len(start_lst)):
            lst_dur_BBP.append(end_lst[i] - start_lst[i])
        return lst_dur_BBP

    def bbp_speech_duration_generator_target_value(self):
        lst_dur_BBP = []
        start_lst, end_lst = self.final_parser()
        for i in range(len(start_lst)):
            lst_dur_BBP.append(end_lst[i] - start_lst[i])
        return sum(lst_dur_BBP)

    def bbp_total_duration_generator_target(self):
        start_lst, end_lst = self.final_parser()
        return end_lst[len(end_lst) - 1] - start_lst[0]

    def bbp_pause_duration_generator_target(self):
        total_duration = self.bbp_total_duration_generator_target()
        speech_duration = self.bbp_speech_duration_generator_target_value()
        return total_duration - speech_duration

    def outlier_detector(self):
        lst_dur_BBP = self.bbp_speech_duration_generator_target()
        if len(lst_dur_BBP) < 2:
            return 0
        else:
            lst_range = []
            lst_threshold_bbp = [i for i in range(len(lst_dur_BBP)) if
                                 lst_dur_BBP[i] < self.THRESH * np.mean(lst_dur_BBP)]
            for i in range(0, len(lst_threshold_bbp) - 1):
                lst_range.append(np.abs(lst_threshold_bbp[i] - lst_threshold_bbp[i + 1]))
            if 1 not in lst_range:
                return 0
            else:
                return lst_threshold_bbp, lst_range

    def parsing_aligner(self):
        if not self.outlier_detector():
            return 0
        start_lst_new, end_lst_new = self.audio_parser()
        lst_threshold_bbp, lst_range = self.outlier_detector()
        for i in reversed(range(len(lst_range))):
            if lst_range[i] == 1:
                idx = lst_threshold_bbp[i]
                start_lst_new.pop(idx + 1)
                end_lst_new.pop(idx)
        return start_lst_new, end_lst_new

    def final_parser(self):
        if self.parsing_aligner():
            return self.parsing_aligner()
        return self.audio_parser()

    def iou_calculator(self):
        import math
        start_lst, end_lst = self.start_end_list_generator()
        if not self.parsing_aligner():
            start_lst_new, end_lst_new = self.final_parser()
        else:
            start_lst_new, end_lst_new = self.parsing_aligner()
        iou_list = []
        for jik in range(0, min(len(start_lst), len(end_lst), len(start_lst_new), len(end_lst_new))):
            union_cal = (max(end_lst[jik], end_lst_new[jik]) - min(start_lst[jik], start_lst_new[jik]))
            intersection_cal = (min(end_lst[jik], end_lst_new[jik]) - max(start_lst[jik], start_lst_new[jik]))
            iou_list.append(abs(intersection_cal / union_cal) * 100)
        return iou_list

class DTWImplementation:
    def __init__(self, dist):
        self.dist = dist
    # dtw algorithm based on drop-dtw paper
    def dtw_paper(self):

        """Classical DTW algorithm"""
        nrows, ncols = self.dist.shape
        # nrows, ncols = len(dist1), len(dist2)
        dtw = np.zeros((nrows + 1,ncols + 1), dtype=np.float32)
        # get dtw table
        for i in range(0, nrows + 1):
            for j in range(0, ncols + 1):
                if (i == 0) and (j == 0):
                    new_val = 0.0
                    dtw[i, j] = new_val
                elif (i == 0) and (j != 0):
                    new_val = np.inf
                    dtw[i, j] = new_val
                elif (i != 0) and (j == 0):
                    new_val = np.inf
                    dtw[i, j] = new_val
                else:
                    neighbors = [dtw[i, j - 1], dtw[i - 1, j - 1], dtw[i - 1, j]]
                    new_val = self.dist[i - 1, j - 1] + min(neighbors)
                    dtw[i, j] = new_val
        # get alignment path
        path = self.traceback(dtw)
        return dtw, path

    def traceback(self, d):
        i, j = np.array(d.shape) - 2
        p, q = [i], [j]
        while (i > 0) or (j > 0):
            tb = np.argmin((d[i, j], d[i, j + 1], d[i + 1, j]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:  # (tb == 2):
                j -= 1
            p.insert(0, i)
            q.insert(0, j)
        return np.array(p), np.array(q)


class AudioAligner:
    def __init__(self, SPEECH_FILE, x2, fs2, counts_num_max, hop_size=512):
        self.SPEECH_FILE = SPEECH_FILE
        self.x2 = x2
        self.fs2 = fs2
        self.counts_num_max = counts_num_max
        self.hop_size = hop_size

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def sequence_matching(self, X, Y, subseq=False):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Perform some shape-squashing here
        # Put the time axes around front
        X = np.swapaxes(X, -1, 0)
        Y = np.swapaxes(Y, -1, 0)

        # Flatten the remaining dimensions
        # Use F-ordering to preserve columns
        X = X.reshape((X.shape[0], -1), order="F")
        Y = Y.reshape((Y.shape[0], -1), order="F")

        try:
            C = distance.cdist(X, Y, metric="euclidean")
        except ValueError as exc:
            raise ParameterError(
                "scipy.spatial.distance.cdist returned an error.\n"
                "Please provide your input in the form X.shape=(K, N) "
                "and Y.shape=(K, M).\n 1-dimensional sequences should "
                "be reshaped to X.shape=(1, N) and Y.shape=(1, M)."
            ) from exc

        # for subsequence matching:
        # if N > M, Y can be a subsequence of X
        if subseq and (X.shape[0] > Y.shape[0]):
            C = C.T

        return C

    def audio_aligner(self):
        ref_audio = reference_audio_dict[self.counts_num_max]
        ref_parse = reference_parse_dict[self.counts_num_max]

        x_1, fs1 = librosa.load(ref_audio)
        # x_2, fs2 = librosa.load(self.SPEECH_FILE)

        ref_audio = reference_audio_dict[self.counts_num_max]
        ref_parse = reference_parse_dict[self.counts_num_max]

        sound = AudioSegment.from_file(ref_audio, "wav")
        normalized_sound = self.match_target_amplitude(sound, -65.0)
        normalized_sound.export("ref_audio.wav", format="wav")

        sound_target = AudioSegment.from_file(self.SPEECH_FILE)
        normalized_sound_target = self.match_target_amplitude(sound_target, -65.0)
        normalized_sound_target.export("target_audio.wav", format="wav")

        x_1, fs1 = librosa.load("ref_audio.wav")
        self.x2, self.fs2 = librosa.load("target_audio.wav")

        # make and display a mel-scaled power (energy-squared) spectrogram
        S1 = librosa.feature.melspectrogram(y=x_1, sr=fs1, n_mels=128)
        S2 = librosa.feature.melspectrogram(y=self.x2, sr=self.fs2, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S1 = librosa.amplitude_to_db(S1)
        log_S2 = librosa.amplitude_to_db(S2)
        x_1_mfcc = librosa.feature.mfcc(S=log_S1, n_mfcc=13)
        x_2_mfcc = librosa.feature.mfcc(S=log_S2, n_mfcc=13)

        matrix_mfcc = self.sequence_matching(x_1_mfcc, x_2_mfcc)
        D, wp = DTWImplementation(matrix_mfcc).dtw_paper()
        wp = np.asarray(wp)
        wp = wp.T
        wp_s = np.asarray(wp) * self.hop_size / fs1

        return x_1, self.x2, fs1, self.fs2, ref_parse, wp, wp_s

