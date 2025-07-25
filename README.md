# Introduction

**This repo is specifically designed for the deployment of multilingual ASR models, while also being compatible with standard monolingual ASR models.**

# Installation(with CUDA)
*If you have installed sherpa-onnx before, please uninstall it first.*
```bash
git clone git@github.com:yangb05/sherpa-onnx.git
cd sherpa-onnx
mkdir build
cd build

cmake \
  -DSHERPA_ONNX_ENABLE_PYTHON=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_GPU=ON \
  ..

make -j
export PYTHONPATH=$PWD/../sherpa-onnx/python/:$PWD/lib:$PYTHONPATH
```
To check that sherpa-onnx has been successfully installed, please use:
```bash
python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
```
It should print some output like below:
```bash
/Users/fangjun/py38/lib/python3.8/site-packages/sherpa_onnx/__init__.py
```
If you want to install the CPU version of sherpa-onnx, please refer to [this tutorial](https://k2-fsa.github.io/sherpa/onnx/python/install.html#method-4-for-developers).

# Deployment(with CUDA)
To deploy the model, you first need to export it to the ONNX format. Refer to __ for the export method. 
Additionally, the corresponding `tokens.txt` file for the model is required, which is generated during the training of the BPE model. 
If you want to start a secure WebSocket server, you can run `sherpa-onnx/python-api-examples/web/generate-certificate.py` to generate the certificate `cert.pem`.

In general, after deploying the model, It is necessary to test whether the deployment was successful. Therefore, it is recommended to provide an audio file in the corresponding language for testing.

After preparing all the required files, you can start a transducer based streaming ASR service like this:
```bash
export CUDA_VISIBLE_DEVICES=6
python sherpa-onnx/python-api-examples/streaming_server.py \
  --encoder encoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --decoder decoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --joiner joiner-epoch-25-avg-15-chunk-16-left-128.onnx \
  --tokens tokens.txt \
  --doc-root sherpa-onnx/python-api-examples/web \
  --port 50351 \
  --provider cuda \
  --certificate sherpa-onnx/python-api-examples/web/cert.pem
```
The started ASR service can be tested like this:
```bash
python sherpa-onnx/python-api-examples/online-websocket-client-decode-file-cert.py \
  --server-addr localhost \
  --server-port 50351 \
  --langtag '<VI>' \
  test_audio.wav
```
`<VI>` is the langtag for Vietnamese，the langtags for other languages ares:
 - `<ZH>`, Chinese
 - `<EN>`, English
 - `<VI>`, Vietnamese
 - `<RU>`, Russian
 - `<JA>`, Japanese
 - `<AR>`, Arabic
 - `<TH>`, Thai
 - `<ID>`, Indonisian


# Сборка для GO

Сборка библиотеки в PowerShell:

```shell
# Make 64 bit
pyenv local 3.11.3
$env:SHERPA_ONNX_CMAKE_ARGS = "-A x64"
python3 setup.py build --plat-name win-amd64

cp -R ./build/sherpa_onnx ./build/sherpa_onnx_amd64
rm ./build/sherpa_onnx -r -Force

pyenv local 3.11.3-win32
$env:SHERPA_ONNX_CMAKE_ARGS = "-A Win32"
python3 setup.py build --plat-name win32

cp -R ./build/sherpa_onnx ./build/sherpa_onnx_win32
rm ./build/sherpa_onnx -r -Force
```


Релиз обертки GO в bash:

```shell
cd ./scripts/go

git clone git@github.com:lichnost/sherpa-onnx-go-windows.git
cp -v ./sherpa_onnx.go ./sherpa-onnx-go-windows/
cp -v ../../sherpa_onnx/c-api/c-api.h ./sherpa-onnx-go-windows

rm -fv sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu/*
dst=$(realpath sherpa-onnx-go-windows/lib/x86_64-pc-windows-gnu)

cp -v ../../build/sherpa_onnx_win32/bin/*.dll $dst
cp -v ../../build/sherpa_onnx_win32/bin/*.lib $dst


rm -fv sherpa-onnx-go-windows/lib/i686-pc-windows-gnu/*
dst=$(realpath sherpa-onnx-go-windows/lib/i686-pc-windows-gnu)

cp -v ../../build/sherpa_onnx_amd64/bin/*.dll $dst
cp -v ../../build/sherpa_onnx_amd64/bin/*.lib $dst

echo "------------------------------"
cd sherpa-onnx-go-windows
git status
git add .
git commit -m "Release v$SHERPA_ONNX_VERSION" && \
git push && \
git tag v$SHERPA_ONNX_VERSION && \
git push origin v$SHERPA_ONNX_VERSION || true
cd ..
rm -rf sherpa-onnx-go-windows
```
