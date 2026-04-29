import spaces
import gradio as gr
from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
from f5_tts.api import F5TTS
import tempfile
import os
f5tts = F5TTS()


@spaces.GPU
def run_tts(ref_audio, ref_text, gen_text, remove_silence=False):
    output_wav_path = tempfile.mktemp(suffix=".wav")

    wav, sr, _ = f5tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=output_wav_path,
        remove_silence=remove_silence,
    )

    return output_wav_path

demo = gr.Interface(
    fn=run_tts,
    inputs=[
        gr.Audio(label="Reference Audio", type="filepath"),
        gr.Textbox(label="Reference Text", placeholder="some call me nature, others call me mother nature."),
        gr.Textbox(label="Generation Text", placeholder="I don't really care what you call me..."),
        gr.Checkbox(label="Remove Silence from Output?", value=False)
    ],
    outputs=gr.Audio(label="Generated Speech"),
    title="🗣️ F5-TTS Demo",
    description="Upload a reference voice, give reference and generation text, and hear it in the same voice!",
)

if __name__ == "__main__":
    demo.launch()
