FROM python:3.8.6-slim-buster
# ENV http_proxy http://proxy-chain.xxx.com:911/ 
# ENV https_proxy http://proxy-chain.xxx.com:912/ 

WORKDIR /

# install dependencies:
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#set the time zone to New York, Same as US Markets:
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#Copy source Code
COPY /app .

CMD ["python", "-u","live_trader.py"]