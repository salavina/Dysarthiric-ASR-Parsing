# custom library import
from helper import ASRLoader, AudioParser, AudioAligner

PATH_TO_VOCAB_FILE = "./neurological_dataset_all_types_plus_TTS_set3/vocab.json"
PATH_TO_WEIGHTS_FOLDER = "./neurological_dataset_all_types_plus_TTS_set3/checkpoint-4850/"

# sample audio file (to be replaced by the audio recorded on alphawit front-end)
SPEECH_FILE = './NF00_10_01_BBP_NORMAL.wav'

counts_num_max, transcription = ASRLoader(PATH_TO_VOCAB_FILE, PATH_TO_WEIGHTS_FOLDER).apply_ml(SPEECH_FILE)

x_1, x_2, fs1, fs2, ref_parse, wp, wp_s = AudioAligner(SPEECH_FILE, counts_num_max).audio_aligner()

obj = AudioParser(fs1, fs2, ref_parse, path=wp)


print("Transcription: ", transcription)
print("Total Number of Repetitions: ", counts_num_max)
print("Total Duration: ", obj.bbp_total_duration_generator_target())
print("Speech Duration: ", obj.bbp_speech_duration_generator_target_value())
print("Pause Duration: ", obj.bbp_pause_duration_generator_target())
# speaking rate: total # of words spoken / total duration * 60 (since speaking rate is # of words spoken in 60 seconds)
print("Speaking Rate: ", len(obj.final_parser()[0])*60/obj.bbp_total_duration_generator_target())