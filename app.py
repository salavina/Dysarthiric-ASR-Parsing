from flask import Flask, jsonify
from flask import request, redirect
from utils import ASRLoader, AudioParser, AudioAligner
import soundfile as sf
import json

PATH_TO_VOCAB_FILE = "./neurological_dataset_all_types_plus_TTS_set3/vocab.json"
PATH_TO_WEIGHTS_FOLDER = "./neurological_dataset_all_types_plus_TTS_set3/checkpoint-4850/"

output_json = {"transcription": "",
               "counts_num_max": 0,
               "total_duration": 0.0,
               "speech_duration": 0.0,
               "pause_duration": 0.0,
               "speaking_rate": 0.0}


def convert_float_to_str(x: float) -> str:
    return f"{x:.2f}"


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            # audio, sr = sf.read(file)
            output_json['counts_num_max'], output_json['transcription'] = ASRLoader(PATH_TO_VOCAB_FILE,
                                                                                    PATH_TO_WEIGHTS_FOLDER).apply_ml(
                file.filename)
            x_1, x_2, fs1, fs2, ref_parse, wp, wp_s = AudioAligner(file.filename,
                                                                   output_json['counts_num_max']).audio_aligner()
            obj = AudioParser(fs1, fs2, ref_parse, path=wp)
            output_json['total_duration'] = "{:.2f}".format(obj.bbp_total_duration_generator_target())
            output_json['speech_duration'] = "{:.2f}".format(obj.bbp_speech_duration_generator_target_value())
            output_json['pause_duration'] = "{:.2f}".format(obj.bbp_pause_duration_generator_target())
            output_json['speaking_rate'] = "{:.2f}".format(len(obj.final_parser()[0]) * 60 / obj.bbp_total_duration_generator_target())
    # return json
    return jsonify(output_json)


if __name__ == "__main__":
    app.run(debug=True)
