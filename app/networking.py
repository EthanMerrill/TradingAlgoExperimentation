#%%
# this file handles all the communications with APIS such as alpaca and google cloud
# general imports 
import requests
import pandas as pd
import datetime as dt
from helper_functions import ensure_dir, log_traceback
#google cloud imports
import io
from io import BytesIO
from google.cloud import storage
import math
#Alpaca keys handling imports
import os
import json
import alpaca_trade_api as tradeapi

try:
    with open('GOOGLE_APPLICATION_CREDENTIALS.json') as f:
        GACdata = json.load(f)

    with open('ALPACA_KEYS.json') as m:
        ALPACA_DATA = json.load(m)

    with open('POLYGON_API.json') as o:
        POLYGON_API = json.load(o)

    #these keys must be set before calling fastquant3 file
    os.environ["alpaca_secret_paper"] = ALPACA_DATA["alpaca_secret_paper"]
    os.environ["alpaca_secret_live"] = ALPACA_DATA["alpaca_secret_live"]
    os.environ["alpaca_live"] = ALPACA_DATA["alpaca_live"]
    os.environ["alpaca_paper"] = ALPACA_DATA["alpaca_paper"]

    os.environ['polygon'] = POLYGON_API['api_key']

    
except Exception as e:
    print(f"Error loading keys from google key manager: error {e}")

  # Set varables depending on paper trading or not
PAPER_TRADE = True

class alpaca_api:
    def __init__(self, PAPER_TRADE):
        if PAPER_TRADE==True:
            self.api_base = 'https://paper-api.alpaca.markets'
            self.headers = {
            "APCA-API-KEY-ID": os.environ["alpaca_paper"], 
            "APCA-API-SECRET-KEY": os.environ["alpaca_secret_paper"]
            }
        elif PAPER_TRADE==False:
            self.api_base ="https://api.alpaca.markets"
            self.headers = {
            "APCA-API-KEY-ID": os.environ["alpaca_live"],
            "APCA-API-SECRET-KEY": os.environ["alpaca_secret_live"]
            }
    def create_api(self):
        api = tradeapi.REST(self.headers.get("APCA-API-KEY-ID"), self.headers.get("APCA-API-SECRET-KEY") , base_url=self.api_base)
        return api
#%%
## Create a object to handle polygon data requests
class polgon_data:
    def __init__(self):
        self.alpaca_live = os.environ["alpaca_live"]
    def get_single_stock_daily_bars(self, symbol, start_date, end_date):
        #takes date times in dt fromat, then converts to strings which are inserted in the api request
        start_date = dt.datetime.strftime(start_date, "%Y-%m-%d")
        end_date = dt.datetime.strftime(end_date.today(), "%Y-%m-%d")
        return requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?unadjusted=false&sort=asc&apiKey={self.alpaca_live}").json().get("results")


# Create a cloud object which can upload to positions or backtests easily
class cloud_object:
    def __init__(self, BUCKET_NAME):
        # Setup Storage client
        self.storage_client = storage.Client.from_service_account_json('GOOGLE_APPLICATION_CREDENTIALS.json')
                # #make bucket object locally:
        self.bucket = self.storage_client.get_bucket(BUCKET_NAME)
    
    
    def save_to_backtests(self, df, blob_name):
        cloud_dir =(f"Backtests/{str(blob_name)}.csv")
        # pd.to_pickle(df,stream)
        self.blob = self.bucket.blob(cloud_dir)
        stream = io.StringIO()
        # df.to_csv(stream, sep=",")
        df.to_csv(stream, header=True, index=True)
        self.blob.upload_from_string(stream.getvalue(),content_type="application/octet-stream")
   
        return (str(blob_name))

    def save_to_positions(self, df, blob_name):
        cloud_dir = (f"Positions/positions-{str(blob_name)}.csv")
        self.blob = self.bucket.blob(cloud_dir)
        stream = io.StringIO()
        df.to_csv(stream, header=True, index=True)
        self.blob.upload_from_string(stream.getvalue(),content_type="application/octet-stream")
        
        return (str(blob_name))

    def download_from_backtests(self, filename):
        cloud_dir = (f"Backtests/{str(filename)}.csv")
        # Check to see if the file exists in the cloud:
        if self.bucket.blob(cloud_dir).exists(self.storage_client) == False:
            raise Exception(f"could not get file from gcloud:'{cloud_dir}.csv'")
        self.blob = self.bucket.blob(cloud_dir)

        raw_bytes = self.blob.download_as_string()
        raw_csv = raw_bytes.decode("utf-8")
        df = pd.read_csv(io.StringIO(raw_csv))
        
        return df

    def download_from_positions(self, filename):
        cloud_dir = (f"Positions/positions-{str(filename)}.csv")
        # Check to see if the file exists in the cloud:
        if self.bucket.blob(cloud_dir).exists(self.storage_client) == False:
            raise Exception(f"could not get file from gcloud:'{cloud_dir}.csv'")
        self.blob = self.bucket.blob(cloud_dir)
        raw_bytes = self.blob.download_as_string()
        raw_csv = raw_bytes.decode("utf-8")
        df = pd.read_csv(io.StringIO(raw_csv))
        
        return df

#%%