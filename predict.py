import tensorflow as tf
import numpy as np
import time
from sense_hat import SenseHat

sense = SenseHat()
sense.clear()

LABELS = ["move_none", "move_circle", "move_shake", "move_twist"]
COLORS = {
    "move_none": [0, 0, 0],
    "move_circle": [255, 0, 0],
    "move_shake": [0, 255, 0],
    "move_twist": [0, 0, 255]
}

interpreter = tf.lite.Interpreter(model_path="motion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def read_imu_sample():
    acc = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()
    return [acc['x'], acc['y'], acc['z'], gyro['x'], gyro['y'], gyro['z']]

try:
    while True:
        print("Collecting 1s sample...")
        samples = [read_imu_sample() for _ in range(50)]
        input_data = np.expand_dims(np.array(samples).flatten().astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = int(np.argmax(output))
        label = LABELS[predicted_index]
        print(f"Predicted: {label}")
        sense.clear(COLORS[label])
        time.sleep(1.0)
except KeyboardInterrupt:
    sense.clear()
