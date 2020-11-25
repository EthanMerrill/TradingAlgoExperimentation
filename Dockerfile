FROM python:3.8.6-slim-buster

#get the environment variables with build arg
ARG GOOGLE_APPLICATION_CREDENTIALS
ARG ALPACA_KEYS
#set environment variables
ENV GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}
ENV alpaca_keys=${ALPACA_KEYS}
ENV PORT 8080


#exposes port 8080 
EXPOSE 8080 


COPY ALPACA_KEYS.json .
COPY GOOGLE_APPLICATION_CREDENTIALS.json .

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