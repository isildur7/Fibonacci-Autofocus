# Fibonacci Focus for Basler and Xavier NX
# Amey Chaware
# November 2020

import time
import A4988 as driver
import baslerwrappers as bsw

import cv2
import numpy as np
from LEDarray import LEDarray

def laplacian(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Laplacian(img_gray,cv2.CV_16U)
    return cv2.mean(img_sobel)[0]

def calculation(camera, conv, fom, position, motor):
    motor.goToPosition(position, 850)
    image = bsw.take_one_opencv_image(camera, conv)
    return fom(image)

def energy_laplacian(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #print(np.shape(img_gray))
    kernel = np.array([[-1, -4, -1],
              [-4, 20, -4],
              [-1, -4, -1]])
    img_sobel = cv2.filter2D(img_gray, cv2.CV_16U, kernel)
    #print(np.shape(img_sobel))
    return np.mean(np.square(img_sobel))

def brenner(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #print(np.shape(img_gray))
    kernel = np.array([-1, 0, 1])
    img_sobel = cv2.filter2D(img_gray, cv2.CV_16U, kernel)
    #print(np.shape(img_sobel))
    return np.sum(np.square(img_sobel))

def normed_variance(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #mean, std = cv2.meanStdDev(img_gray)
    #mean, std = np.squeeze(mean), np.squeeze(std)
    return np.var(img_gray)/np.mean(img_gray)
    #return std*std/mean

def get_fibonacci_list(max_index):
    fibo = [1,1]
    for i in range(2,max_index+1):
        fibo.append(fibo[i-1]+fibo[i-2])
    return fibo

def Fibonacci(start, end, N, camera, converter, motor, fom):
    # N is the least index of Fibonacci Sequence such that F_N >= endofrange - startofrange
    fibo = get_fibonacci_list(N)
    picturePos1 = 0
    picturePos2 = 0
    calculate = 0
    for i in range(1, N):
        scale = fibo[N-i-1]/fibo[N-i+1]
        if i == 1:
            picturePos1 = start + int(scale*(end-start))
            picturePos2 = end - int(scale*(end-start))
            fomAt1 = calculation(camera, converter, fom, picturePos1, motor)
            fomAt2 = calculation(camera, converter, fom, picturePos2, motor)
            
        if calculate == 1:
            fomAt1 = calculation(camera, converter, fom, picturePos1, motor)
        elif calculate == 2:
            fomAt2 = calculation(camera, converter, fom, picturePos2, motor)
        
        if fomAt1 < fomAt2:
            start = picturePos1
            picturePos1 = picturePos2
            picturePos2 = end - int(scale*(end - start))
            fomAt1 = fomAt2
            calculate = 2
        else:
            end = picturePos2
            picturePos2 = picturePos1
            picturePos1 = start + int(scale*(end - start))
            fomAt2 = fomAt1
            calculate = 1
    
    if calculation(camera, converter, fom, picturePos1, motor) > \
        calculation(camera, converter, fom, picturePos2, motor):
        return picturePos1
    else:
        return picturePos2

if __name__ == "__main__":
    # First parameter is CLK and the second one is DIN
    dot = LEDarray(n_arrays = 1, brightness = 0.4)
    #dots.fill((70, 110, 80)) # turn on illumination
    dot.turnAllOff()
    dot.turnOnFillCircle(1,(157,125,138)) 

    # turn on camera
    camera = bsw.create_simple_camera("/home/amey/Desktop/python-basics/Basler/10x-brightfield.pfs")
    conv = bsw.opencv_converter()
    bsw.change_ROI(camera, (1280,960), ((2592-1280)//2, (1944-960)//2))

    # turn on motor
    motor = driver.StepperMotor(motor_pin=16, dir_pin=18, ms3_pin=13, base_mode=driver.EIGHTH_STEP)
    
    # focusing
    print("Start focusing")
    bsw.start_imaging(camera)
    startt = time.time()
    focus_position = Fibonacci(start=0, end=4000, N=18, camera=camera, converter=conv, motor=motor, fom=normed_variance)
    # camera.capture(camera, format='jpeg', bayer=True)
    motor.goToPosition(focus_position, 850)
    endt = time.time()
    bsw.stop_imaging(camera)

    #set camera resolution to 2592x1944
    bsw.max_ROI(camera)
    bsw.start_imaging(camera)
    #save image to file.
    bsw.capture_and_save_png(camera, "/home/amey/Desktop/python-basics/Basler/test-for-focus.png")
    bsw.stop_imaging(camera)
    bsw.close_camera(camera)
    
    dot.turnAllOff()
    
    print(focus_position)
    print(endt-startt)
    print(motor.getCurrentPosition())
    motor.goToPosition(0, 850) #take it back to start
