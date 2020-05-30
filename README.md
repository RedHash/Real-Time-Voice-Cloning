# Voice-Cloning

# Installation (Ubuntu 10)

* sudo apt-get install ffmpeg
* sudo apt-get install lame
* pip3 install deepspeech 

Download [data and pretrained weights](https://drive.google.com/file/d/1J06_TPFiE8U_Eeh1mrY6T635Q36XZ3Jr/view?usp=sharing),
then put it into project folder

* unzip data.zip
* mv pretrained.zip TextToSpeech/.

* cd TextToSpeech
* pip3 install -r requirements.txt
* unzip pretrained.zip

# Usage
```bash
bash change_voice.sh <speech_template_path.mp3> <reference_voice_path.mp3> <output_path.mp3>
```

# Idea

Voice -> Text -> Voice

Use speech recognition module (TextToSpeech -- TTS) from Mozilla ([DeepSpeech architecture](https://github.com/mozilla/DeepSpeech)) to translate 
voice into text, then apply [Real Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) (SpeechToText -- STT)
to synthesize speech with specified voice.

## Advantages
0) IT WORKS!1!1!!1111!!!!
1) Fast-learning -- only a short recording of the necessary voice is needed
2) Sufficient quality with small GPU resources
3) Easy-to-use pipeline

## Disadvantages
1) As far as I understood, TTS module is able to work with punctuation, which STT module doesn't provide. It can reduce speech quaility
2) Not end-to-end pipline, both modules should be trained independently

# Results
Reference recording of Bill Gates [link](https://yadi.sk/d/kPCupGlmbaVSpQ)
1. [My own voice](https://yadi.sk/d/KAw8GTJuBHADFQ)
2. [Professional recording](https://yadi.sk/d/uNiakkdjB3zwWQ)
