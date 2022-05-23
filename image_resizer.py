from os import listdir, mkdir
from os.path import isfile, join, exists
import cv2

mypath = "C:\\Users\\sigtu\\Documents\\tower\\RES\\images\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

if not exists(mypath + "\\transform\\"):
    mkdir(mypath + "\\transform\\")

for files in onlyfiles:
    if ".png" in files:
        image = cv2.imread(mypath + files)
        image = cv2.resize(image, (77, 77), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mypath + "\\transform\\" + files.replace(".png", ".bmp"), image)
