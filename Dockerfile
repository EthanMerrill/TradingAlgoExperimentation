FROM python:3.8.6-buster

WORKDIR /app

# install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy source Code
COPY /app .

CMD ["python", "live_trader.py"]