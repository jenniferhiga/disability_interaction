#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:52:41 2023

@author: jenniferhiga
"""

# Import necessary libraries for Flask
from flask import Flask, render_template, Response
# Import necessary libraries for GazeTracking
import cv2
import numpy as np
import simpleaudio as sa
from gaze_tracking import GazeTracking
# Import necessary libraries for FFT analyser
import argparse
from src.stream_analyzer import Stream_Analyzer
import time

# Initialize the Flask app
app = Flask(__name__)

# Initialise the Gaze Tracker and OpenCV inputs
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


def gaze_tracker():
    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            break
        
        # Process the frame to track the user's gaze
        gaze.refresh(frame)
        new_frame = gaze.annotated_frame()
        
        # Get the coordinates of the right pupil and check for blinking
        right_coords = gaze.pupil_right_coords()
        if gaze.is_blinking():
            print("Blink")

        # Generate a note frequency based on the x coordinate of the right pupil
        try:
            frequency = right_coords[0] / 2
        except TypeError:
            print("Cannot see your eyes")
            continue
        
        # Generate a sine wave with the note frequency and play it using simpleaudio
        fs = 44100
        seconds = 0.1
        t = np.arange(0, seconds, 1/fs)
        note = np.sin(frequency * t * 2 * np.pi)
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        audio = audio.astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        play_obj.wait_done()

        # Add a welcome message to the frame and display it
        text = "Welcome Jenny"
        cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        _, jpeg = cv2.imencode('.jpg', new_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None, dest='device',
                        help='pyaudio (portaudio) device index')
    parser.add_argument('--height', type=int, default=450, dest='height',
                        help='height, in pixels, of the visualizer window')
    parser.add_argument('--n_frequency_bins', type=int, default=400, dest='frequency_bins',
                        help='The FFT features are grouped in bins')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--window_ratio', default='24/9', dest='window_ratio',
                        help='float ratio of the visualizer window. e.g. 24/9')
    parser.add_argument('--sleep_between_frames', dest='sleep_between_frames', action='store_true',
                        help='when true process sleeps between frames to reduce CPU usage (recommended for low update rates)')
    return parser.parse_args()

def convert_window_ratio(window_ratio):
    if '/' in window_ratio:
        dividend, divisor = window_ratio.split('/')
        try:
            float_ratio = float(dividend) / float(divisor)
        except:
            raise ValueError('window_ratio should be in the format: float/float')
        return float_ratio
    raise ValueError('window_ratio should be in the format: float/float')

def run_FFT_analyzer():
    args = parse_args()
    window_ratio = convert_window_ratio(args.window_ratio)
# =============================================================================
#  Jennifer's sound system audio output channels
#     > 0 MacBook Air Microphone, Core Audio (1 in, 0 out)
#     < 1 MacBook Air Speakers, Core Audio (0 in, 2 out)
#       2 Microsoft Teams Audio, Core Audio (2 in, 2 out)
# =============================================================================
    ear = Stream_Analyzer(
                    device = args.device,        # Pyaudio (portaudio) device index, 1 is speakers
                    rate   = None,               # Audio samplerate, None uses the default source settings
                    FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                    updates_per_second  = 1000,  # How often to read the audio stream for new data
                    smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                    n_frequency_bins = args.frequency_bins, # The FFT features are grouped in bins
                    visualize = 1,               # Visualize the FFT features with PyGame
                    verbose   = args.verbose,    # Print running statistics (latency, fps, ...)
                    height    = args.height,     # Height, in pixels, of the visualizer window,
                    window_ratio = window_ratio  # Float ratio of the visualizer window. e.g. 24/9
                    )

    fps = 60  #How often to update the FFT features + display
    last_update = time.time()
    while True:
        if (time.time() - last_update) > (1./fps):
            last_update = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
        elif args.sleep_between_frames:
            time.sleep(((1./fps)-(time.time()-last_update)) * 0.99)
            
@app.route('/')
def hello():
    return "Hello, Jenny"

@app.route('/stream_analyser')
def index():
    return Response(run_FFT_analyzer())

@app.route('/video_feed')
def video_feed():
    return Response(gaze_tracker(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5004)
    
