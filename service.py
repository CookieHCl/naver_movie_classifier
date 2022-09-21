import re
import json
import bentoml
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bentoml.io import Text
from bentoml.io import Text

naver_movie_runner = bentoml.keras.get("naver_movie:latest").to_runner()

svc = bentoml.Service("naver_movie_classifier", runners=[naver_movie_runner])
mecab = Mecab()
with open('tokenizer.json') as f:
	data = json.load(f)
	tokenizer = tokenizer_from_json(data)

@svc.api(input=Text(), output=Text())
def classify(input_series: str) -> str:
	stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
	new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', input_series)
	new_sentence = mecab.morphs(new_sentence)
	new_sentence = [word for word in new_sentence if not word in stopwords]
	encoded = tokenizer.texts_to_sequences([new_sentence])
	pad_new = pad_sequences(encoded, maxlen = 80)

	score = float(naver_movie_runner.predict.run(pad_new))
	if(score > 0.5):
		return ("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
	else:
		return ("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
