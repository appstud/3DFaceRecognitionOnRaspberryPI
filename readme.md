## Introduction

This code is a facial recognition system for  RaspberryPi based on 3D frontalization for face alignment and LBPH for recognition.

It is tested on Python3.6, and it uses dlib, Opencv3.0 and flask (for streaming the output to your web browser)

## Install requirements
```sh
pip install -r requirements.txt
```
## Create your 3D aligned database
### save videos of you and people you want to recognize
To use it you must first create a database of the persons you want to recognize by filming the persons using the raspberry pi's camera (resolution: 320x240) and placing the videos of each person in a directory that has his/her name.
```sh
python save_videos_from_webcam.py --name_of_person my_name
```
### 3D alignment
To build your database of aligned faces run:
```sh
python face_recognition_HAAR.py
```
## Frontalization
The frontalization algorithm is based on the implementation in : https://github.com/dougsouza/face-frontalization It has been changed in order to adapt for facial recognition on a raspberry Pi.

## Test 
To test the algorithm using the raspberry s camera run: 
```sh
python main.py --ip "the ip adress of the raspberry" -o 8000
```
Type the ip adress of your raspberry followed by :8000 in your browser to see the video stream
