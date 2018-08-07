tensorflow-experiments
==========================

I am trying to figure out how much slower tensorflow js is compared to python.
This example trains an LSTM on some house prices data from Zillow.

## Run in node
```
npm install
npm start
```

Results:
```
Parsing data: 1 ms
Initializing model: 316 ms
First inference: 287 ms
Training: 1638640 ms
Inference: 212 ms

[0/100] Average Loss: 0.12670819215815177;
[99/100] Average Loss: 0.00033980431982689134;
```

## Run in the browser
```
npm install
npm run local
```

Navigate to `localhost:8080`, and press the "Train" button. Timings are logged
to the console.

Results:
```
Parsing data: 6 ms
Initializing model: 1564 ms
First inference: 4484 ms
Training: 879413 ms
Inference: 292 ms

[0/100] Average Loss: 0.12691785972988742;
[99/100] Average Loss: 0.0003770285519418725;
```

## Run in python
Warning: I am not a python developer, so if this isn't right, 🤷‍♀️. First, figure
out how to [install tensorflow](https://www.tensorflow.org/install) since
this is a dependency of the project. It will barf errors if you don't have installed
correctly along the lines of "No module named tensorflow". I don't know how to help with that.

Then:

```
python3 run.py
```
