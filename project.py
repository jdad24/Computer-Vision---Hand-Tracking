from handdetector import HandDetector
import sys
import cv2
import pyautogui

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):

	# Initializing the dimensions of the image to be resized and grabbing the image size
	dim = None
	(h, w) = image.shape[:2]

	# If both the width and height are None, then return the original image
	if width is None and height is None:
		return image

	# Checking to see if the width is None
	if width is None:
		# Calculating the ratio of the height and constructing the dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# Otherwise, the height is None
	else:
		# Calculating the ratio of the width and constructing the dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# Resizing the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# Returns the resized image
	return resized

def main():

	if len(sys.argv) != 3:
		print("Usage: python project.py fist2.xml (1 or 2)")
		return

	# Hand Detector is instatntiated with path to the cascade classifier
	hd = HandDetector(sys.argv[1])

	# Read video from webcam
	camera = cv2.VideoCapture(0)

	# Looping over all the frames in the video
	while True:

		# Grabs the current frame
		(grabbed, frame) = camera.read()

		# Resizing the frame and converting it to grayscale
		frame = resize(frame, width = 300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detecting hands in the image and then cloning the frame so that we can draw on it
		handRects = hd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
			minSize = (30, 30))
		frameClone = frame.copy()

		# Looping over the bounding boxes and drawing them
		for (fX, fY, fW, fH) in handRects:
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

			# Mapping mouse movement with hand tracking
			if sys.argv[2] == '1':
				pyautogui.moveTo(fX * 5, fY * 9, 0, pyautogui.easeInOutQuad)
			elif sys.argv[2] == '2':
				if fY > 80:
					pyautogui.scroll(-2)
				elif fY > 60 and fY < 80:
					pyautogui.scroll(-1)
				elif fY < 60 and fY > 40:
					pyautogui.scroll(0)
				elif fY < 40 and fY > 20:
					pyautogui.scroll(1)
				elif fY < 20:
					pyautogui.scroll(2)

		# Showing detected hands
		cv2.imshow("Hands", frameClone)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()