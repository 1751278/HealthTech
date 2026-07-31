import serial
import time
import keyboard

# Replace 'COM4' with the specific port assigned by your OS
# For Mac/Linux use something like '/dev/cu.ESP32_Serial'
COM_PORT = 'COM17' 
BAUD_RATE = 115200

def receive_bluetooth_data():
    print(f"Connecting to ESP32 on {COM_PORT}...")
    try:
        # Open the serial port connection
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Allow connection to settle
        print("Connected successfully! Listening for data...")
        
        while True:
            if ser.in_waiting > 0:
                # Read line, decode bytes to string, and strip extra whitespaces/newlines
                raw_data = ser.readline()
                decoded_data = raw_data.decode('utf-8', errors='ignore').strip()
                print(f"Received: {decoded_data}")

            if keyboard.is_pressed('q'):
                print(" 'q' pressed. Exiting...")
                break
            elif keyboard.is_pressed('r'):
                print("reset position")
                message = "resetPos"
                ser.write(message.encode('utf-8'))

                
    except serial.SerialException as e:
        print(f"Error connecting to serial port: {e}")
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    receive_bluetooth_data()