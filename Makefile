init:
	python -m venv .venv
	.venv\Scripts\pip install -r requirements.txt

run:
	.venv\Scripts\python main.py
