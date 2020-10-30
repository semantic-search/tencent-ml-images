import os
from dotenv import load_dotenv
load_dotenv()
KAFKA_HOSTNAME = os.getenv("KAFKA_HOSTNAME")
KAFKA_PORT = os.getenv("KAFKA_PORT")
RECEIVE_TOPIC = 'TENCENT_ML_IMAGES_MODEL'
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
DB = os.getenv('MONGO_DB')
PORT = os.getenv('MONGO_PORT')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
CLIENT_ID = os.getenv('CLIENT_ID')
DASHBOARD_URL = os.getenv('DASHBOARD_URL')
LOGSTASH_HOSTNAME = os.getenv('LOGSTASH_HOSTNAME')
LOGSTASH_PORT = os.getenv('LOGSTASH_PORT')
PARENT_NAME = 'Image'
GROUP_NAME = 'Image_Recognition'
