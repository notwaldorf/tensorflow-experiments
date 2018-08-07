import tensorflow as tf
import json
import time

current_time = lambda: int(round(time.time() * 1000))

# Got data from https:#www.zillow.com/research/data/
# 1 bedroom time series, by city.
Data = None
Model = None

LEARNING_RATE = 0.001
EPOCHS = 100
MAX_LOSS = 0.00000001
UNITS = 60

def main():
    with open("./data.json") as f:
        obj = json.load(f)
        data_received(obj)
        train()

def data_received(obj):
    start = current_time()
    Data = parse_data(obj)
    print("Parsing data: %d ms" % (current_time() - start))

    start = current_time()
    init_model()
    print("Initializing model: %d ms" % (current_time() - start))

    start = current_time()
    # prediction = getNormalizedPrediction(Data.trainingData[0])
    print("First inference: %d ms" % (current_time() - start))

def parse_data(obj):
    # The data is:
    # - 11 entries
    # - columns: RegionName/State/SizeRank/1996-04/1996-05...

    # Names of cities
    cities = []
    dates = []
    prices = []

    # min/max price
    minPrice = 100000
    maxPrice = -1

    ignoredCols = ["RegionName", "State", "SizeRank"]

    for entry in obj:
        # Save this just in case we need it.
        cities.append(entry['RegionName'])

        theseCols = entry.keys()
        # If this entry doesn't have any new columns, move on.
        if len(ignoredCols) + len(dates) != len(theseCols):
            # Go through all the columns, and save all the ones that are
            # the dates (i.e. aren't names) and we haven't seen before
            for column in theseCols:
                if column not in ignoredCols and column not in dates:
                    dates.append(column)

        # Save the prices for this city. We're assuming not all cities have all the dates.
        values = []
        for date in dates:
          values.append(entry[date])

        prices.append(values)
        minPrice = min(minPrice, min(values))
        maxPrice = max(maxPrice, max(values))

    parsed = dict()
    parsed["cities"] = cities
    parsed["dates"] = dates
    parsed["prices"] = prices
    parsed["minPrice"] = minPrice
    parsed["maxPrice"] = maxPrice

    return parsed

def init_model():
    print("ok")


def train():
    print('Start training...')

if __name__ == '__main__':
  main()
