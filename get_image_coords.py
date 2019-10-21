import argparse
import cv2
 
 
def click_and_crop(event, x, y, flags, param):
	refPt = []
 
	# if the left mouse button was clicked, print (x,y)
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		print(refPt)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break
 
cv2.destroyAllWindows()