import tensorflow as tf
import json
import time
import numpy as np

current_time = lambda: int(round(time.time() * 1000))

LEARNING_RATE = 0.001
EPOCHS = 100
MAX_LOSS = 0.00000001
UNITS = 60

def main():
    # Got data from https:#www.zillow.com/research/data/
    # 1 bedroom time series, by city.
    with open("./data.json") as f:
        obj = json.load(f)

        start = current_time()
        data = parse_data(obj)
        print("Parsing data: %d ms" % (current_time() - start))

        start = current_time()
        model = init_model(data)
        print("Initializing model: %d ms" % (current_time() - start))

        start = current_time()
        print("Initial predictions:")
        for i in range(len(data["trainingData"])):
            prediction = get_normalized_prediction(model, data["trainingData"][i], data["minPrice"], data["maxPrice"])
            y = denormalize(data["trainingData"][i]["output"], data["minPrice"], data["maxPrice"])
            print("%f | %f" % (prediction, y))
        print("Inference: %d ms" % (current_time() - start))

        train(model, data)

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

def init_model(data):
    # Normalize the data.
    data["trainingData"] = get_training_data(data)

    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(units=UNITS),
      tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam', loss="mean_squared_error", learning_rate=LEARNING_RATE)
    return model

def get_training_data(data):
    trainingData = []
    for i, city in enumerate(data["cities"]):
        ys = data["prices"][i]
        num_ys = len(ys) - 1
        normalized = []

        for j in range(num_ys):
            normalized.append(normalize(ys[j], data["minPrice"], data["maxPrice"]))

        last = normalize(ys[num_ys], data["minPrice"], data["maxPrice"])

        #input = tf.constant(normalized, shape = [1, len(normalized), 1])
        #output = tf.constant(last,  shape = [1, 1])
        trainingData.append({"input":normalized, "output":last})
    return trainingData

def train(model, data):
    start = current_time()
    training_data = data["trainingData"];
    for e in range(EPOCHS):
        totalLoss = 0
        for i in range(len(training_data)):
            _input = np.array(training_data[i]["input"])
            _input = np.reshape(_input, (1, 266, 1))
            output = np.array(training_data[i]["output"])
            output = np.reshape(output, (1, 1))

            info = model.fit(
              x=_input, y=output, epochs=1, verbose=0)
            totalLoss += info.history["loss"][0]

        avgLoss = totalLoss / 266.0
        print("[%d/%d] Average Loss %.9f" % (e, EPOCHS, avgLoss))

    print("Training: %d ms" % (current_time() - start))

    start = current_time()
    print("Final predictions:")
    for i in range(len(data["trainingData"])):
        prediction = get_normalized_prediction(model, data["trainingData"][i], data["minPrice"], data["maxPrice"])
        y = denormalize(data["trainingData"][i]["output"], data["minPrice"], data["maxPrice"])
        print("%f | %f" % (prediction, y))
    print("Inference: %d ms" % (current_time() - start))
    # print(model.summary())

def get_normalized_prediction(model, data, minPrice, maxPrice):
  #prediction = model.predict(data["input"], steps=1).flatten()
  prediction = model.predict(tf.constant(data["input"], shape=[1, len(data["input"]), 1]), steps=1).flatten()
  # prediction = prediction.get(0,0)
  return denormalize(prediction, minPrice, maxPrice)

# Converts values to the range between values 0 and 1
def normalize(num, min, max):
	return (num - min) * (1/(max - min))

# Reconverts values from range between values 0 and 1 to range between Min and Max
def denormalize(num, min, max):
	return (num / (1/(max - min))) + min

if __name__ == '__main__':
  main()
