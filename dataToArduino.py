import time
import json
import serial
try:
    bluetoothSerial = serial.Serial( "/dev/rfcomm{bt}".format(bt=4), baudrate=9600 )
except NameError:
    print("Please connect to the Bluetooth")
while(True):
    time.sleep(0.2)
    try:
        with open('data2.txt','r') as json_file:
            data = json.load(json_file)
        #print(list(data.values()))
        values_list = list(data.values())
        nothing_msg = str(values_list[0])
        bottle_msg = str(values_list[1])
        msg = nothing_msg + bottle_msg
        print(msg)
        if msg == '01':
            flag_bottle = 101
            j = str('P')
            b = j.encode()
            bluetoothSerial.write(b)
            time.sleep(0.1)
        elif msg == '10':
            j = str('N')
            b = j.encode()
            bluetoothSerial.write(b)
            time.sleep(0.1)
    except:
        import testin