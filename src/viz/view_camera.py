import pyrealsense2 as rs, numpy as np, cv2

dev = rs.context().devices[0]
print(dev.get_info(rs.camera_info.name),
      "/ USB", dev.get_info(rs.camera_info.usb_type_descriptor))

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)   # half the bandwidth
pipe.start(cfg)
try:
    while True:
        f = pipe.wait_for_frames(10000).get_color_frame()
        if not f: continue
        cv2.imshow("RealSense (q to quit)", np.asanyarray(f.get_data()))
        if cv2.waitKey(1) & 0xFF == ord("q"): break
finally:
    pipe.stop(); cv2.destroyAllWindows()