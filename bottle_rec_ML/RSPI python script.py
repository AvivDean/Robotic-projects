import numpy as np
import cv2
import tensorflow as tf
import threading
import json
import edgeDetection
import RPi.GPIO as GPIO
#set GPIO pins
GPIO.setmode(GPIO.BCM)
#blue
GPIO.setup(27,GPIO.OUT)
GPIO.output(27,False)
#red
GPIO.setup(22,GPIO.OUT)
GPIO.output(22,False)
lock = threading.Lock()
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
labels = ['bottle','can','nothing']
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
bottle_counter = 0
bottle_state = 0
cap = cv2.VideoCapture(0)

def main():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        b = cv2.resize(frame,(224,224),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        # Our operations on the frame come here
        gray = cv2.cvtColor(b,1)
        normalized_image_array = (gray.astype(np.float32) / 127.0) - 1
        normalized_image_array.resize([1,224,224,3])
        input_data = normalized_image_array
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data
        bottle = prediction[0,0]
        cans = prediction[0,1]
        nothing = prediction[0,2]
        if nothing > 0.6:
            print('nothing')
            nothing_state = 1
            GPIO.output(27,True)
        else:
            nothing_state = 0
            GPIO.output(27,False)
        if bottle > 0.6:
            print('bottle')
            bottle_state = 1
            bottle_counter = bottle_counter+1
            GPIO.output(22,True)
        if bottle_counter > 20:
            cv2.imwrite("./Captures/bottle.jpg",frame)
            edgeDetection.main()
            bottle_counter = 0
        else:
            GPIO.output(22,False)
            bottle_state = 0
            bottle_counter = 0
            lock.acquire()
            data = {"nothing_state":nothing_state,"bottle_state":bottle_state}
        with open('data2.txt','w') as outfile:
            json.dump(data,outfile)
            lock.release()
            # Display the resulting frame
            cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            GPIO.output(22,False)
            GPIO.output(27,False)
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
main()