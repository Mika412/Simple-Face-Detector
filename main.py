from matplotlib import pyplot
import cv2
from numpy import asarray
import numpy as np
from PIL import Image
import os, os.path
from scipy.spatial.distance import cosine
import onnxruntime as rt
from datetime import datetime

class FaceDetector:
	def __init__(self):
		self.sess = rt.InferenceSession("models/model.onnx")
		for idx, inp in enumerate(self.sess.get_inputs()):
			print("Input info: {}".format(inp))
		self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
		self.load_people()

		self.previous_frame_faces = []


	def preprocess_input(self, x, data_format=None, version=1):
		x_temp = np.copy(x)
		x_temp = x_temp[..., ::-1]
		x_temp[..., 0] -= 91.4953
		x_temp[..., 1] -= 103.8827
		x_temp[..., 2] -= 131.0912

		return x_temp
	def extract_faces(self, image, required_size=(224, 224)):
		
		startTime = datetime.now()
		faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
	
		if len(faces) == 0:
			return [], []

		face_array = []
		for (x,y,w,h) in faces:
			x2, y2 = x + w, y + h
			face = image[y:y2, x:x2]
			img = Image.fromarray(face)
			img = img.resize(required_size)
			face_array.append(asarray(img))
		return faces, face_array


	# extract faces and calculate face embeddings for a list of photo files
	def get_embeddings(self, faces):
		#faces = [].append(faces)
		faces = np.expand_dims(faces, axis=0)
		# convert into an array of samples
		samples = asarray(faces, 'float32')
		# prepare the face for the model, e.g. center pixels
		samples = self.preprocess_input(samples)
		yhat = self.sess.run(None, {"input_1": samples})
		return yhat[0]

	def fast_scandir(self, dir):
		subfolders= [f.path for f in os.scandir(dir) if f.is_dir()]
		for dir in list(subfolders):
			subfolders.extend(self.fast_scandir(dir))
		return subfolders


	def is_match(self, known_embeddings, candidate_embedding, thresh=0.5):
		# calculate distance between embeddings
		scores = []
		for known_embedding in known_embeddings:
			scores.append(cosine(known_embedding, candidate_embedding))
		score = np.average(scores)
		return score, score <= thresh

	def load_people(self):
		peoplePaths = self.fast_scandir("images")
		self.people = []
		for path in peoplePaths:
			imgs = []
			valid_images = [".jpg",".gif",".png",".tga"]
			for f in os.listdir(path):
				ext = os.path.splitext(f)[1]
				if ext.lower() not in valid_images:
					continue
				imgs.append(pyplot.imread(os.path.join(path,f)))
			embeddings = []
			for img in imgs:
				_, faces =  self.extract_faces(img)
				for face in faces:
					embeddings.append(self.get_embeddings(face))
			
			print(path + " " + str(len(embeddings)))
			self.people.append({'Name': path, 'Embeddings': embeddings})

	def overlap_area(self, x1, y1, h1, w1, x2,y2, h2,w2):  # returns None if rectangles don't intersect
		dx = min(x1+w1, x2+w2) - max(x1, x2)
		dy = min(x1+h1, x2+h2) - max(y1, y2)
		if (dx>=0) and (dy>=0):
			return dx*dy / (h1*w1)
	def intersection(self,a,b):
		x = max(a[0], b[0])
		y = max(a[1], b[1])
		w = min(a[0]+a[2], b[0]+b[2]) - x
		h = min(a[1]+a[3], b[1]+b[3]) - y
		if w<0 or h<0: return (0,0,0,0), 0 # or (0,0,0,0) ?
		return (x, y, w, h), (w*h)/(a[2]*a[3])

	def detectFace(self, img):
		face_cord, faces =  self.extract_faces(img)
		
		detected_people = []
		unknown_faces = []
		for i in range(len(faces)):
			(x,y,w,h) = face_cord[i]
			has_detected = False
			detected_name = ""
			for j in range(len(self.previous_frame_faces)):
				(x2, y2, w2, h2, detected_name) = self.previous_frame_faces[j]
				_, int_perc = self.intersection((x,y,w,h),(x2,y2,w2,h2))

				if int_perc > 0.6:
					has_detected = True
					break
			if has_detected:
				detected_people.append((x,y,w,h, detected_name))
			else:
				unknown_faces.append(i)
		
		for i in unknown_faces:
			(x,y,w,h) = face_cord[i]
			embedding = self.get_embeddings(faces[i])

			potential_people = []
			for person in self.people:
				score, isMatch = self.is_match(person['Embeddings'], embedding)
				if isMatch:
					potential_people.append([person['Name'], score])
			if len(potential_people) > 0:
				detected_people.append((x,y,w,h, potential_people[0][0][7:]))
			else:
				detected_people.append((x,y,w,h, 'Unknown'))
				
		print(detected_people)

		self.previous_frame_faces = detected_people
		for i in range(len(detected_people)):
			(x,y,w,h, name) = detected_people[i]
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
		return img

def main():
	faceDetector = FaceDetector()

	cap = cv2.VideoCapture(0)
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		img2 = faceDetector.detectFace(img)
		img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
		# Display the resulting frame
		cv2.imshow('Face detector',img2)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	
if __name__== "__main__":
	main()
