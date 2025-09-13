# send_mask_with_meta.py
import json, time, zmq, numpy as np
WIDTH, HEIGHT = 1920, 1080
mask = np.zeros((HEIGHT, WIDTH), np.uint8); 
mask[590:690, 970:1150] = 255
meta = {"mask_id":2,"width":WIDTH,"height":HEIGHT,"channels":1,"dtype":"uint8","sent_unix_ns":time.time_ns()}
ctx = zmq.Context.instance(); 
s = ctx.socket(zmq.PUSH); 
s.setsockopt(zmq.LINGER, 0); 
s.connect("tcp://127.0.0.1:5556")
s.send_multipart([json.dumps(meta).encode("utf-8"), memoryview(mask)], copy=False)
print("sent")
