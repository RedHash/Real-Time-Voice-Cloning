import numpy as np
import librosa
import argparse
import torch
import sys
from tqdm import tqdm

from TextToSpeech.encoder import model_embedding_size as speaker_embedding_size
from TextToSpeech.utils.argutils import print_args
from TextToSpeech.synthesizer.inference import Synthesizer
from TextToSpeech.encoder import inference as encoder
from TextToSpeech.vocoder import inference as vocoder
from pathlib import Path

if __name__ == '__main__':
    # Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="TextToSpeech/encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path,
                        default="TextToSpeech/synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="TextToSpeech/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="TextToSpeech/vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("-r", "--reference_audio", type=Path,
                        default="reference.mp3",
                        help="Path to a reference audio")
    parser.add_argument("-t", "--text_to_gen", type=Path,
                        default="text.txt",
                        help="Path to a text to be spoken")
    parser.add_argument("--low_mem", action="store_true", help= \
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help= \
        "If True, audio won't be played.")
    args = parser.parse_args()
    print_args(args, parser)

    if not args.no_sound:
        import sounddevice as sd

    # Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))

    # Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    in_fpath = args.reference_audio

    ## Computing the embedding
    # First, we load the wav using the function that the speaker encoder provides. This is
    # important: there is preprocessing that must be applied.

    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    print("Ref. audio is loaded")

    # Then we derive the embedding. There are many functions and parameters that the
    # speaker encoder interfaces. These are mostly for in-depth research. You will typically
    # only use this function (with its default parameters):
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")

    text_file = open(args.text_to_gen)
    texts = text_file.read().split('\n')
    embeds = [embed]

    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    print("Gen. spectrograms")
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    
    spec = specs[0]  # TODO : check this out

    # Generating the waveform
    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)

    # Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Play the audio (non-blocking)
    if not args.no_sound:
        sd.stop()
        sd.play(generated_wav, synthesizer.sample_rate)

    # Save it on the disk
    fpath = "demo_output_%02d.wav" % i
    librosa.output.write_wav(fpath, generated_wav.astype(np.float32),
                             synthesizer.sample_rate)