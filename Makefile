init:
	python -m venv venv
	.\venv\Scripts\activate
	pip install -r requirements.txt

run:
	python main.py
