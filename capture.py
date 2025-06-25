from sense_hat import SenseHat
import time, os, numpy as np

LABEL = "move_circle"
SAMPLES = 50
FREQ_HZ = 50
DELAY = 1.0 / FREQ_HZ
save_dir = f"./motion_data/{LABEL}"
os.makedirs(save_dir, exist_ok=True)
sense = SenseHat()

while True:
    input("Press Enter to record 1 second...")
    data = []
    for _ in range(SAMPLES):
        acc = sense.get_accelerometer_raw()
        gyro = sense.get_gyroscope_raw()
        data.append([acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']])
        time.sleep(DELAY)
    np.save(f"{save_dir}/{LABEL}_{int(time.time())}.npy", np.array(data))
