"""
Needs to be generalised and made shorter 
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import functools
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
import pandas as pd
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt


class CustomMainWindow(QMainWindow):
    def __init__(self):
        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("my first window")
        
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        
        # Place the zoom button
        self.zoomBtn = QPushButton(text = 'zoom')
        self.zoomBtn.setFixedSize(100, 50)
        self.zoomBtn.clicked.connect(self.zoomBtnAction)
        self.LAYOUT_A.addWidget(self.zoomBtn, *(0,0))
        
        # Place the matplotlib figure
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))
        
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        self.show()
        return

    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)
        return

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.myFig.addData(value)
        return

''' End Class '''


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        self.addedData = []
        print(matplotlib.__version__)
        # The data
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50
        # The window
        self.fig = Figure(figsize=(5,5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(755,765)
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)
        return

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])
        return

    def addData(self, value):
        self.addedData.append(value)
        return

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()
        return

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]
        return

''' End Class '''


class Communicate(QObject):
    data_signal = pyqtSignal(float)

''' End Class '''


def dataSendLoop(addData_callbackFunc):
    """ Callback function for signal acquisiton """

    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    sampling_rate = 320 # Basically this is the sampling rate 
    # y = get_data(sampling_rate)
    y = daq_stream(arduino, n_data=sampling_rate, delay=8)

    i = 0

    while(True):
        if(i > sampling_rate-1):
            i = 0
        time.sleep(0.1)
        mySrc.data_signal.emit(y[i]) # <- Here you emit a signal!
        i += 1
   

def get_data(len):
    """ Insert a random singal emitter here to see it plotted live (also change dataSendLoop) """
    n = np.linspace(0, 499, 500) # Return evenly spaced numbers over a specified interval.
    return (50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5)))
  

# Basically we have to embed live streaming data in the data acquistion loop to see if it works
# I really like the black box theory



""" ARDUINO SECTION """

HANDSHAKE = 0
VOLTAGE_REQUEST = 1
ON_REQUEST = 2
STREAM = 3
READ_DAQ_DELAY = 4

def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.manufacturer is not None and "Arduino" in p.manufacturer:
                port = p.device
    return port


def handshake_arduino(
    arduino, sleep_time=1, print_handshake_message=False, handshake_code=0
):
    """Make sure connection is established by sending
    and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 2

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout

def parse_raw(raw):
    """Parse bytes output from Arduino."""
    raw = raw.decode()
    if raw[-1] != "\n":
        raise ValueError(
            "Input must end with newline, otherwise message is incomplete."
        )

    t, V = raw.rstrip().split(",")

    return int(t), int(V) 


def daq_stream(arduino, n_data=125, delay=8):
    """Obtain `n_data` data points from an Arduino stream
    with a delay of `delay` milliseconds between each."""

    # Specify delay
    arduino.write(bytes([READ_DAQ_DELAY]) + (str(delay) + "x").encode())

    # Initialize output
    time_ms = np.empty(n_data)  
    voltage = np.empty(n_data)

    # Turn on the stream
    arduino.write(bytes([STREAM]))

    # Receive data
    i = 0
    while i < n_data:
        raw = arduino.read_until()

        try:
            t, V = parse_raw(raw)
            time_ms[i] = t
            voltage[i] = V
            i += 1
        except:
            pass

    # Turn off the stream
    # arduino.write(bytes([ON_REQUEST]))
    # We want it to be an infinity loop to be terminated by the user

    return voltage



# Main call

if __name__== '__main__':
    
    # Arduino streaming start up
    port = find_arduino()
    arduino = serial.Serial(port, baudrate=115200)
    handshake_arduino(arduino, handshake_code=HANDSHAKE, print_handshake_message=True)

    # PyQt plotting start up
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())

"""
References - Live Plotting Code and arduino streaming to python
1) K.Mulier
    https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
2) https://be189.github.io/lessons/13/streaming_data_from_arduino.html
"""



