bash run_stt.sh $1
bash run_tts.sh $2
lame output.wav
mv output.mp3 $3
rm output.wav
