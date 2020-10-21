import cv2
import numpy
# from matplotlib import pyplot as plt
import time
import collections
import random

USE_IMAGE_NOT_FOUND_EXCEPTION = False
GRAYSCALE_DEFAULT = False
unicode = str
RUNNING_CV_2 = cv2.__version__[0] < '3'
if RUNNING_CV_2:
    LOAD_COLOR = cv2.CV_LOAD_IMAGE_COLOR
    LOAD_GRAYSCALE = cv2.CV_LOAD_IMAGE_GRAYSCALE
else:
    LOAD_COLOR = cv2.IMREAD_COLOR
    LOAD_GRAYSCALE = cv2.IMREAD_GRAYSCALE
    
Box = collections.namedtuple('Box', 'left top width height')
Point = collections.namedtuple('Point', 'x y')
RGB = collections.namedtuple('RGB', 'red green blue')
    
def _load_cv2(img, grayscale=None):
    """
    TODO
    """
    # load images if given filename, or convert as needed to opencv
    # Alpha layer just causes failures at this point, so flatten to RGB.
    # RGBA: load with -1 * cv2.CV_LOAD_IMAGE_COLOR to preserve alpha
    # to matchTemplate, need template and image to be the same wrt having alpha
    # print("len",len(img))
    
    if len(img) > 9:
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # print(img.shape)
        img_cv = img
        # print(">9")
        return img_cv
    
    if grayscale is None:
        grayscale = GRAYSCALE_DEFAULT
        # print("1")

    if isinstance(img, (str, unicode)):
        # The function imread loads an image from the specified file and
        # returns it. If the image cannot be read (because of missing
        # file, improper permissions, unsupported or invalid format),
        # the function returns an empty matrix
        # http://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
        # print("1")

        if grayscale:
            img_cv = cv2.imread(img, LOAD_GRAYSCALE)
            # print("2")

        else:
            img_cv = cv2.imread(img, LOAD_COLOR)
            # print(img_cv.shape)
            # print("3")

        if img_cv is None:
            # print("4")

            raise IOError("Failed to read %s because file is missing, "
                        "has improper permissions, or is an "
                        "unsupported or invalid format" % img)
    elif isinstance(img, numpy.ndarray):
        # print("5")

        # don't try to convert an already-gray image to gray
        if grayscale and len(img.shape) == 3:  # and img.shape[2] == 3:
            img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_cv = img
    elif hasattr(img, 'convert'):
        # print("6")

        # assume its a PIL.Image, convert to cv format
        img_array = numpy.array(img.convert('RGB'))
        img_cv = img_array[:, :, ::-1].copy()  # -1 does RGB -> BGR
        if grayscale:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        # print("7")
        
        raise TypeError('expected an image filename, OpenCV numpy array, or PIL image')
    return img_cv



def locateAll_opencv(needleImage, haystackImage, grayscale=None, limit=10000, region=None, step=1,
                      confidence=0.999):
    """
    TODO - rewrite this
        faster but more memory-intensive than pure python
        step 2 skips every other row and column = ~3x faster but prone to miss;
            to compensate, the algorithm automatically reduces the confidence
            threshold by 5% (which helps but will not avoid all misses).
        limitations:
          - OpenCV 3.x & python 3.x not tested
          - RGBA images are treated as RBG (ignores alpha channel)
    """
    if grayscale is None:
        grayscale = GRAYSCALE_DEFAULT

    confidence = float(confidence)

    needleImage = _load_cv2(needleImage, grayscale)
    needleHeight, needleWidth = needleImage.shape[:2]
    haystackImage = _load_cv2(haystackImage, grayscale)

    if region:
        haystackImage = haystackImage[region[1]:region[1]+region[3],
                                      region[0]:region[0]+region[2]]
    else:
        region = (0, 0)  # full image; these values used in the yield statement
    if (haystackImage.shape[0] < needleImage.shape[0] or
        haystackImage.shape[1] < needleImage.shape[1]):
        # avoid semi-cryptic OpenCV error below if bad size
        raise ValueError('needle dimension(s) exceed the haystack image or region dimensions')

    if step == 2:
        confidence *= 0.95
        needleImage = needleImage[::step, ::step]
        haystackImage = haystackImage[::step, ::step]
    else:
        step = 1

    # get all matches at once, credit: https://stackoverflow.com/questions/7670112/finding-a-subimage-inside-a-numpy-image/9253805#9253805
    result = cv2.matchTemplate(haystackImage, needleImage, cv2.TM_CCOEFF_NORMED)
    match_indices = numpy.arange(result.size)[(result > confidence).flatten()]
    matches = numpy.unravel_index(match_indices[:limit], result.shape)

    if len(matches[0]) == 0:
        if USE_IMAGE_NOT_FOUND_EXCEPTION:
            raise ImageNotFoundException('Could not locate the image (highest confidence = %.3f)' % result.max())
        else:
            return

    # use a generator for API consistency:
    matchx = matches[1] * step + region[0]  # vectorized
    matchy = matches[0] * step + region[1]
    for x, y in zip(matchx, matchy):
        yield Box(x, y, needleWidth, needleHeight)


from importlib import import_module
import os

import win32gui
import win32con
import win32ui
import win32api

import re
from time import sleep


def FindWindow_bySearch(pattern):
    window_list = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), window_list)
    for each in window_list:
        if re.search(pattern, win32gui.GetWindowText(each)) is not None:
            return each

def getWindow_W_H(hwnd):
    # 取得目標視窗的大小
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    width = right - left - 15
    height = bot - top - 11
    return (left, top, width, height)

def getWindow_Img(hwnd):
    # 將 hwnd 換成 WindowLong
    s = win32gui.GetWindowLong(hwnd,win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, s|win32con.WS_EX_LAYERED)
    # 判斷視窗是否最小化
    show = win32gui.IsIconic(hwnd)
    # 將視窗圖層屬性改變成透明    
    # 還原視窗並拉到最前方
    # 取消最大小化動畫
    # 取得視窗寬高
    if show == 1: 
        win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 0)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_ALPHA)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)    
        x, y, width, height = getWindow_W_H(hwnd)        
    # 創造輸出圖層
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    # 取得視窗寬高
    x, y, width, height = getWindow_W_H(hwnd)
    # 如果視窗最小化，則移到Z軸最下方
    if show == 1: win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, x, y, width, height, win32con.SWP_NOACTIVATE)
    # 複製目標圖層，貼上到 bmp
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0 , 0), (width, height), srcdc, (8, 3), win32con.SRCCOPY)
    # 將 bitmap 轉換成 numpy
    signedIntsArray = bmp.GetBitmapBits(True)
    img = numpy.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4) #png，具有透明度的
    # 釋放device content
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    # 還原目標屬性
    if show == 1 :
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
        win32gui.SystemParametersInfo(win32con.SPI_SETANIMATION, 1)
    # 回傳圖片
    return img



import win32gui
import win32con
import win32api
import time
import os

times = 6

handle = win32gui.FindWindow(None,'BlueStacks')
handle = win32gui.FindWindowEx(handle, 0, None, "BlueStacks Android PluginAndroid")
hwnd = handle
if handle == 0:
    for i in range(10):
       print("沒有捕捉到視窗")
else:
    left, top, right, bot = win32gui.GetWindowRect(handle)  # 視窗所在位置的坐標
    # for t in range(5):
    #     times -= 1
    #     print('將在倒數%d秒後點擊現在滑鼠所在的位置' % times)
    #     tempt = win32api.GetCursorPos()  # 記錄滑鼠所處位置的坐標
    #     windowRec = win32gui.GetWindowRect(handle)  # 目標子句柄視窗的坐標
    #     x = tempt[0] - windowRec[0]  # 計算相對x坐標
    #     y = tempt[1] - windowRec[1]  # 計算相對y坐標
    #     print('坐標為', x, y)
    #     time.sleep(1)  # 每1s輸出一次

def doClick(cx, cy):#點擊坐標
    # print('點擊',x,y,'坐標')
    long_position = win32api.MAKELONG(cx, cy)  # 模擬滑鼠指針 傳送到指定坐標
    win32api.SendMessage(handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)  # 模擬滑鼠按下
    time.sleep(random.uniform(0.1, 0.2))
    win32api.SendMessage(handle, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, long_position)  # 模擬滑鼠彈起

# if x>=left and y>=top and x < right and y < bot:
#     print('5秒後點擊坐標',x,y)
#     time.sleep(5)  # 每5s輸出一次
#     doClick(x, y)  # 可以後台點擊但是程式視窗不能最小化
# elif x > 9999 and y > 9999:
#     for i in range(10):
#         print('程式視窗不能最小化')
#         break
# else:
#     for i in range(10):
#         print('滑鼠不在目標視窗界面')
#         break     

while True:
    # hwnd = FindWindow_bySearch("BlueStacks")
    img = getWindow_Img(hwnd)
    points = tuple(locateAll_opencv('ready.png',img,confidence=0.8))
    if len(points) > 0:
        # print(points[0])
        x=points[0].left
        y=points[0].top
        w=points[0].width
        h=points[0].height
        doClick(x,y)
        print("拋竿")
        # time.sleep(random.uniform(1, 3))    
        while True:
            img = getWindow_Img(hwnd)
            points = tuple(locateAll_opencv('ok.png',img,confidence=0.9))
            if len(points) > 0:
                # print(points[0])
                x=points[0].left
                y=points[0].top
                w=points[0].width
                h=points[0].height
                doClick(x,y)
                print("收竿")
                break
        print("開始釣下一條魚")
        time.sleep(random.uniform(1, 3))

        
