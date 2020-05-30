#!/bin/bash
ffmpeg -i $1 -acodec pcm_s16le -ar 16000 text.wav

deepspeech --model SpeechToText/deepspeech-0.7.0-models.pbmm  --audio text.wav --extended > text.txt

rm text.wav
