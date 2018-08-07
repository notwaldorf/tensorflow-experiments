import tensorflow as tf
import json

# Got data from https://www.zillow.com/research/data/
# 1 bedroom time series, by city.
# Data = {};
# Model = {};

LEARNING_RATE = 0.001;
EPOCHS = 100;
MAX_LOSS = 0.00000001;
UNITS = 60;

def main():
    with open('./data.json') as f:
        obj = json.load(f)
        data_received()
        train()

def data_received():
    print('in data received')

def train():
    print('Start training...')

if __name__ == '__main__':
  main()
