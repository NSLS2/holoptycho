#!/usr/bin/env python3
# NOTE: decoding currently works only with bslz4 encoding. lz4 encoding does not work !!!

import time
import zmq
import random
import json
import pprint
import numpy as np
import traceback
from dectris.compression import decompress

eiger_ip = "10.66.19.45"
#eiger_ip = "localhost"
eiger_port = "5559"

n_messages = 0



def consumer():
    global n_messages
    context = zmq.Context()
    # recieve work
    consumer_receiver = context.socket(zmq.PULL)
    consumer_receiver.connect(f"tcp://{eiger_ip}:{eiger_port}")

    data_encoding, data_shape, data_type, next_is_frame = None, None, None, False
    n_frame = 0
    print(f"Waiting for data ...")

    while True:
        try:
            msg = None
            msg = consumer_receiver.recv(flags=zmq.NOBLOCK)
        except:
            #print("No data, wait 0.1s")
            time.sleep(0.1)
        #n_messages += 1
        #print(f"Message {n_messages} is received")

        if msg:
            try:
                if not next_is_frame:
                    buffer = json.loads(msg.decode())
                    
                    # There should be more robust way to detect this frame
                    if "shape" in buffer:
                        data_h = buffer.get("htype",None)
                        data_encoding = buffer.get("encoding", None)
                        data_shape = buffer.get("shape", None)
                        data_type = buffer.get("type", None)
                        next_is_frame = True

                    result = buffer

                else:
                    next_is_frame = False
                    
                    try:
                        supported_encodings = {"bs32-lz4<": "bslz4", "bs16-lz4<": "bslz4", "lz4<": "lz4"}
                        data_encoding_str = supported_encodings.get(data_encoding, None)
                        #if not data_encoding_str:
                        #    raise RuntimeError(f"Encoding {data_encoding!r} is not supported")
                        
                        supported_types = {"uint32": "uint32",  "uint16": "uint16"}
                        data_type_str = data_type
                        #data_type_str = supported_types.get(data_type, None)
                        if not data_type_str:
                            raise RuntimeError(f"Encoding {data_type!r} is not supported")

                        elem_type = getattr(np, data_type_str)
                        print(elem_type)
                        elem_size = elem_type(0).nbytes
                        print(data_encoding)
                        if data_encoding:
                            decompressed = decompress(msg, data_encoding_str, elem_size=elem_size)
                        else:
                            decompressed = msg
                        
                        data = np.frombuffer(decompressed, dtype=elem_type)
                        data = np.reshape(data, np.flip(data_shape))
                        # The data should be properly shaped image of the respective type
                        print("Data frame")
                        #if 'flat' in data_h:
                        #    np.save('flat.npy',data)
                        #    with open('arr','wb') as f:
                        #        f.write(msg)
                        #if 'image' in data_h:
                        #    np.save('img.npy',data)
                        if 'dimage' in data_h:
                            n_frame += 1
                            print("total frame recv",n_frame)
                        result = f"Data frame is received. Image shape: {data.shape}"

                    except Exception as ex:
                        result = f"Failed to decode the received data frame. The frame size is {len(buffer)} bytes"
                        print(traceback.format_exc())

            except Exception as ex:
                result = f"ERROR: Failed to process message: {ex}"
                #print(msg)
                print(traceback.format_exc())

            print(f"{pprint.pformat(result)}")
            print("=" * 80)

consumer()
