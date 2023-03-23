from flask import Flask, render_template
from flask import request, redirect
from helper import ASRLoader, AudioParser, AudioAligner
import soundfile as sf

app = Flask(__name__, template_folder='templates')

PATH_TO_VOCAB_FILE = "./neurological_dataset_all_types_plus_TTS_set3/vocab.json"
PATH_TO_WEIGHTS_FOLDER = "./neurological_dataset_all_types_plus_TTS_set3/checkpoint-4850/"
def convert_float_to_str(x: float) -> str:
    return f"{x:.2f}"
@app.route("/", methods=["POST", "GET"])
def index():
    transcription = ""
    counts_num_max = 0
    total_duration = 0.0
    speech_duration = 0.0
    pause_duration = 0.0
    speaking_rate = 0.0
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file:
            audio, sr = sf.read(file)
            counts_num_max, transcription = ASRLoader(PATH_TO_VOCAB_FILE, PATH_TO_WEIGHTS_FOLDER).apply_ml(audio, sr)
            x_1, x_2, fs1, fs2, ref_parse, wp, wp_s = AudioAligner(file.filename, audio, sr, counts_num_max).audio_aligner()
            obj = AudioParser(fs1, fs2, ref_parse, path=wp)
            total_duration = obj.bbp_total_duration_generator_target()
            speech_duration = obj.bbp_speech_duration_generator_target_value()
            pause_duration = obj.bbp_pause_duration_generator_target()
            speaking_rate = len(obj.final_parser()[0])*60/obj.bbp_total_duration_generator_target()
# return json
    return render_template("upload.html", text = transcription,
                           counts_num_max = str(counts_num_max),
                           total_duration = str(total_duration),
                           speech_duration = str(speech_duration),
                           pause_duration = str(pause_duration),
                           speaking_rate = str(speaking_rate))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)