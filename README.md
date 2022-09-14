# Tello-Computer-Vision
A series of python modules to control a tello dron with computer vision features.

I´m perfectly aware that many of the modules and general approaches allow (and deserve) a refactorization, but this started as a quick and dirty project and become bigger than expected initially. 

My programming background also comes from other languages, so please be indulgent with my python coding style.
Anyway, if you find this code at least a nice starting point to develop your own projects, I´ll be more than happy.


This project allows testing different use cases of a Tello drone. 
The different use cases are handled through the command line parameters of the TKC script.
Check the TKC implementation to better understand the different options.

The required mode is controlled through the -m _mode_ flag
The parameters also include a -t flag: When activated, no command is actually sent to the drone but the rest of processing is done. Output shows current operations and general info.
The parameters also include a -l flag: This allows to use the laptop camera (or any camera device attached) for those use cases that require usage of camera.

The list of use cases is as follows:

1. Keyboard control (run TKC.py -m k): 
   Lets you control the drone with the keyboard. 
   Implemented in the _keyControl_ method of the _telloControl.py_ module.
2. Control through command (run TKC.py -m c)
   Lets you specify a list of commands that are sequentially sent to the drone.
   Useful in more advanced scenarios where you want the drone to move autonomously following a path.
   Implemented in the _commandControl_ method of the _telloControl.py_ module
3. Search a color (or more properly said, __calibrate__ values for a color) (run TKC.py -m s)
   Used to calibrate the values used for tracking an object identified by its color.
   Implemented in the _calibrate_ method of the _tellocCamera.py_ module 
4. Track an object (run TKC.py -m obj)
   Tracks an object using color segmentation. 
   The function used to recognize the target can be changed.
   Implemented in the _startVideoLoopTarget_ method of the _tellocCamera.py_ module 
   Note that in the constructor of the telloCamera class, the processing function has been set to 'object'
5. Track a face (run TKC.py -m face).
   Tracks a face using face recognition.
   Uses the same method _startVideoLoopTarget_ method of the _tellocCamera.py_ module, but
   in the constructor of the telloCamera class, the processing function has been set to "Face"
6. Search a pattern using a programmed path (run TKC.py -m p)
   Looks for a set of patterns following a pre-programmed path.
   Implemented in the _startVideoLoopSearchShapes_ method of the _tellocCamera.py_ module
7. Recognize a number signed raising fingers and find a text with the given number following a path (run TKC.py -m r)
   Implemented in the _startVideoLoopSearchHand_ method of the _tellocCamera.py_ module
8. Count faces (approach 1 - in beta - try to count by taking several camera shots) (run TKC.py -m mf) in 360 degrees.
   Makes the drone to turn a full 360 degrees and try to count the number of faces.
   Implemented in the _startVideoLoopMultiface_ method of the _tellocCamera.py_ module
9. Count faces (approach 2 - in beta - try to build a panorama picture and then count faces) (run TKC.py -m pan) in 360 degrees.
   Makes the drone to turn a full 360 degrees and try to count the number of faces.
   Implemented in the _startVideoLoopMultiface_ method of the _tellocCamera.py_ module

I've tried to document the code enough with comments, so to understand a feature, 
start with the proper method indicated in the above paragraph and then drill down.


