# USAGE
# Start the server:
# 	python server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@{name image}.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary package
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import sys
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_keras_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = load_model('tl-with-inception.h5')

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(150, 150))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			prediction = preds[0]
			print(preds, file=sys.stdout)
			if (prediction[0] == 1):
				label = 'Bread'
			elif (prediction[1] == 1):
				label = 'Dairy product'
			elif (prediction[2] == 1):
				label = 'Dessert'
			elif (prediction[3] == 1):
				label = 'Egg'
			elif (prediction[4] == 1):
				label = 'Fried food'
			elif (prediction[5] == 1):
				label = 'Meat'
			elif (prediction[6] == 1):
				label = 'Noodles or Pasta'
			elif (prediction[7] == 1):
				label = 'Rice'
			elif (prediction[8] == 1):
				label = 'Seafood'
			elif (prediction[9] == 1):
				label = 'Soup'
			elif (prediction[10] == 1):
				label = 'Vegetable or Fruit'
			elif (prediction[11] == 1):
				label = 'Kue Dadar Gulung'
			elif (prediction[12] == 1):
				label = 'Kastengel'
			elif (prediction[13] == 1):
				label = 'Klepon'
			elif (prediction[14] == 1):
				label = 'Kue lapis'
			elif (prediction[15] == 1):
				label = 'Kue lumpur'
			elif (prediction[16] == 1):
				label = 'Kue putri salju'
			elif (prediction[17] == 1):
				label = 'Risoles'
			elif (prediction[18 == 1]):
				label = 'Serabi'
			r = {"classification": label}
			data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_keras_model()
	app.run()