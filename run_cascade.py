from palmtree.model import load_model
import cv2 as cv
from palmtree.cascade import contains_palmtree, Image, palmtree_markers
import pylab


print("loading image ...")
filename = "pt_hd.jpg"
#filename = "pexels-photo-88212.png"
img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
assert(img is not None)

print("loading model ...")
mod = load_model("wicked.mod")

print("evaluating ...")
image = Image(img)
contains = contains_palmtree(mod, image)
print("contains: %s" % contains)
rects = []
palmtree_markers(rects, mod, image)
print("rects: %s" % len(rects))
 
print("drawing ...")
img_orig = cv.imread(filename)
for rect in rects:
    rect.draw_on(img_orig)
pylab.imshow(img_orig)
pylab.show()
 
print("done.")
