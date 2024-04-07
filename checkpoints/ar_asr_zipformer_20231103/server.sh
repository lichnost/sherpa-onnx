export CUDA_VISIBLE_DEVICES=6
python ../../python-api-examples/streaming_server.py \
  --encoder encoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --decoder decoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --joiner joiner-epoch-25-avg-15-chunk-16-left-128.onnx \
  --tokens tokens.txt \
  --doc-root ../../python-api-examples/web \
  --port 50351 \
  --provider cuda \
  --certificate ../../python-api-examples/web/cert.pem