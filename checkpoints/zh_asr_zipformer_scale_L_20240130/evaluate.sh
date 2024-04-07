wav_dir=/mgData3/yangbo/sherpa-onnx/data/wenetspeech_test_meeting/wav
for wav in $wav_dir/* 
do
  python3 ../../python-api-examples/online-websocket-client-decode-file-cert-eval.py \
    --server-addr localhost \
    --server-port 50152 \
    --sound-file $wav \
    --result-file /mgData3/yangbo/sherpa-onnx/checkpoints/zh_asr_zipformer_scale_L_20240130/streaming.hyp
done