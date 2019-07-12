# Import OpenCV
import cv2

# Will perform hand detection
class HandDetector:
	
	def __init__(self, handCascadePath):
		# Loading the hand detector classifier into memory
		self.handCascade = cv2.CascadeClassifier(handCascadePath)

	def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)):
		# Detects hands in the image
		rects = self.handCascade.detectMultiScale(image,
			scaleFactor = scaleFactor, minNeighbors = minNeighbors,
			minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

		# Returns the bounding boxes
		return rects