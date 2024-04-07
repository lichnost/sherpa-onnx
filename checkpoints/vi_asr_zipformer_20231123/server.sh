export CUDA_VISIBLE_DEVICES=6
python ../../python-api-examples/streaming_server.py \
  --encoder ./encoder-epoch-23-avg-6-chunk-16-left-128.onnx \
  --decoder ./decoder-epoch-23-avg-6-chunk-16-left-128.onnx \
  --joiner ./joiner-epoch-23-avg-6-chunk-16-left-128.onnx \
  --tokens ./tokens.txt \
  --doc-root ../../python-api-examples/web \
  --port 50051 \
  --provider cuda \
  --certificate ../../python-api-examples/web/cert.pem