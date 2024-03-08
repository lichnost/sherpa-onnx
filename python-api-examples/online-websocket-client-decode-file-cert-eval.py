#!/usr/bin/env python3
#
# Copyright (c)  2023  Xiaomi Corporation

"""
A websocket client for sherpa-onnx-online-websocket-server

Usage:
    ./online-websocket-client-decode-file.py \
      --server-addr localhost \
      --server-port 6006 \
      --seconds-per-message 0.1 \
      --samples-per-message 8000 \
      /path/to/foo.wav

(Note: You have to first start the server before starting the client)

You can find the c++ server at
https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-websocket-server.cc
or use the python server ./python-api-examples/streaming_server.py

There is also a C++ version of the client. Please see
https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/online-websocket-client.cc
"""

import argparse
import asyncio
import json
import logging
import wave
import ssl


try:
    import websockets
except ImportError:
    print("please run:")
    print("")
    print("  pip install websockets")
    print("")
    print("before you run this script")
    print("")

import numpy as np


def read_wave(wave_filename: str) -> np.ndarray:
    """
    Args:
      wave_filename:
        Path to a wave file. Its sampling rate has to be 16000.
        It should be single channel and each sample should be 16-bit.
    Returns:
      Return a 1-D float32 tensor.
    """

    with wave.open(wave_filename) as f:
        assert f.getframerate() == 16000, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=6006,
        help="Port of the server",
    )

    parser.add_argument(
        "--samples-per-message",
        type=int,
        default=8000,
        help="Number of samples per message",
    )

    parser.add_argument(
        "--seconds-per-message",
        type=float,
        default=0.06,
        help="We will simulate that the duration of two messages is of this value",
    )

    parser.add_argument(
        "--sound-file",
        type=str,
        required=True,
        help="The input sound file. Must be wave with a single channel, 16kHz "
        "sampling rate, 16-bit of each sample.",
    )
    
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="The result file.",
    )

    return parser.parse_args()


async def receive_results(socket: websockets.WebSocketServerProtocol):
    last_message = ""
    async for message in socket:
        if message != "Done!":
            last_message = message
            logging.info(json.loads(message))
        else:
            break
    return last_message


def preprocess(text):
    # remove space
    text = text.replace(' ', '')
    # remove <sil>
    text = text.replace('<sil>', '')
    # remove puncs
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    dicts={i:'' for i in punctuation}
    punc_table=str.maketrans(dicts)
    text=text.translate(punc_table)
    return text


async def run(
    server_addr: str,
    server_port: int,
    wave_filename: str,
    result_filename: str,
    samples_per_message: int,
    seconds_per_message: float,
):
    data = read_wave(wave_filename)
    key = wave_filename.rsplit('/', 1)[1].split('.')[0]
    use_ssl = True
    if use_ssl:
        cafile = "/mgData3/yangbo/sherpa-onnx/python-api-examples/web/selfsigned.crt"
        certfile = "/mgData3/yangbo/sherpa-onnx/python-api-examples/web/cert.pem"
        keyfile = "/mgData3/yangbo/sherpa-onnx/python-api-examples/web/private.key"
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=cafile)
        ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        uri = "wss://{}:{}".format(server_addr, server_port)
    else:
        uri = "ws://{}:{}".format(server_addr, server_port)
        ssl_context = None
    # logging.info("connect to", uri)
    async with websockets.connect(uri, ssl=ssl_context) as websocket:
        logging.info(f"Sending {wave_filename}")

        receive_task = asyncio.create_task(receive_results(websocket))

        start = 0
        while start < data.shape[0]:
            end = start + samples_per_message
            end = min(end, data.shape[0])
            d = data.data[start:end].tobytes()

            await websocket.send(d)

            # Simulate streaming. You can remove the sleep if you want
            await asyncio.sleep(seconds_per_message)  # in seconds

            start += samples_per_message

        # to signal that the client has sent all the data
        await websocket.send("Done")

        decoding_results = await receive_task
        result = json.loads(decoding_results)
        text = preprocess(result['text'])
        # logging.info(f"\nFinal result is:\n{json.loads(decoding_results)}")
        with open(result_filename, "a") as f:
            f.write(f"{key} {text}\n")



async def main():
    args = get_args()
    logging.info(vars(args))

    server_addr = args.server_addr
    server_port = args.server_port
    samples_per_message = args.samples_per_message
    seconds_per_message = args.seconds_per_message

    await run(
        server_addr=server_addr,
        server_port=server_port,
        wave_filename=args.sound_file,
        result_filename=args.result_file,
        samples_per_message=samples_per_message,
        seconds_per_message=seconds_per_message,
    )


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"  # noqa
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    asyncio.run(main())
