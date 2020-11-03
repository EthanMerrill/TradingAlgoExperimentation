FROM python:3.8.6-slim-buster
# ENV http_proxy http://proxy-chain.xxx.com:911/ 
# ENV https_proxy http://proxy-chain.xxx.com:912/ 

WORKDIR /app

# install dependencies:
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#Copy source Code
COPY /app .

CMD ["python", "-u","live_trader.py"]