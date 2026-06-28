import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import kagglehub
import tensorflow_text as text
import tf_keras as keras
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import json

MODEL_VERSION = "1.0.0"

model = None
model_load_error = None
model_path = None
id_to_team = None

def get_model():
	global model, model_load_error, model_path, id_to_team
	if model is not None:
		return model
	if model_load_error is not None:
		raise RuntimeError(model_load_error)
	try:
		model_path = kagglehub.model_download("pernavjain/customer-support-dispatcher/keras/default")
		print("Path to model files:", model_path)
		model = keras.models.load_model(
			f"{model_path}/customer_support_ticket_dispatcher_v1.keras",
			custom_objects={"KerasLayer": hub.KerasLayer},
			compile=False
		)
		with open(f"{model_path}/id_to_team.json") as f:
			id_to_team = json.load(f)
		return model
	except Exception as e:
		model_load_error = str(e)
		raise


def predict_output(email:list):
	global id_to_team

	loaded_model = get_model()
	prediction = loaded_model.predict(email)

	team_probs = tf.nn.softmax(prediction[0][0], axis=-1).numpy()
	predicted_class = np.argmax(team_probs)
	predicted_team = id_to_team[f"{predicted_class}"]
	confidence = team_probs[predicted_class]

	urgency_score = prediction[1][0][0]
	if (urgency_score<0):
		urgency_score=0
	return {
		'predicted_team' : predicted_team,
		'confidence' : round(float(confidence), 3),
		'urgency_score' : round(float(urgency_score), 3)
	}