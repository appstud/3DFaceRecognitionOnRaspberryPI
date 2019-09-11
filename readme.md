This code is a facial recognition system for a RaspberryPi based on 3D frontalization for face alignment and LBPH for recognition.

It is tested on Python3.6, and it uses dlib, Opencv3.0 and flask (for streaming the output to your web browser)

To use it you must first create a database of the persons you want to recognize by filming the persons using the raspberry pi's camera (resolution: 320,240) and placing the videos of each person in a directory that has his/her name. All these directories must be inside a directory that you name Database in the root folder

The frontalization algorithm is based on the implementation in : https://github.com/dougsouza/face-frontalization
It has been changed in order to adapt for facial recognition on a raspberry Pi.

To build your database of aligned faces run:
python face_recognition_HAAR.py 

To test the algorithm using the raspberry s camera run:
python main.py --ip "the ip adress of the raspberry" -o 8000

Type the ip adress of your raspberry followed by :8000 in your browser to see the video stream
