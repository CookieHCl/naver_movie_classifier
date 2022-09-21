import bentoml
from tensorflow.keras.models import load_model

loaded_model = load_model('best_model.h5')
bentoml.keras.save_model("naver_movie", loaded_model)
