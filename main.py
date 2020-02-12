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

	def detectFace(self, img):
		face_cord, faces =  self.extract_faces(img)

		for i in range(len(faces)):
			(x,y,w,h) = face_cord[i]
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			embedding = self.get_embeddings(faces[i])

			potential_people = []
			for person in self.people:
				score, isMatch = self.is_match(person['Embeddings'], embedding)
				if isMatch:
					potential_people.append([person['Name'], score])
			if len(potential_people) > 0:
				cv2.putText(img, potential_people[0][0][7:], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
			else:
				cv2.putText(img, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
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
