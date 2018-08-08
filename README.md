tensorflow-experiments
==========================

I am trying to figure out how much slower `tensorflow.js` (or `node`) is compared to `python`.

This example trains an LSTM on some house prices data from Zillow, using an LSTM -- I didn't build the model from
scratch, and I used either the prebuilt one from `tensorflow.js` or from
the Keras model in `python`). Training is done for 100 epochs.

I tried to
keep the code between js/python as similar as possible, which means the python
code might look a little suspect (this is also helped by the fact I've basically never written python until this code)

## Results tl; dr
The training data is historical house prices for the top 11 cities on Zillow (267 data points for each city). I held back the last data point as the `output`, and used the other 266 for the LSTM `input`. The LSTM has 60 hidden units, and is trained for 100 epochs (1 episode per epoch).

All the tests were ran on the same MacBook Pro.

|  | Initializing model (ms) | Total training (ms)  | Inference (ms)|
| ------------- | -------------------:| ---------:| ----------:|
| Python        | 40 | 250627 | 2924 |
| Node          | 664 | 3905974 | 4453 |
| Browser       | 1445 | 1382144 | 2924 |

Training looks 5.5 times faster in Python than in the browser, and 16 times faster than in node 🙀.

## Prediction values
(Each row represents one of the 11 cities. The "actual value" is
the `output` point used in training; the remaining values are the predictions for the sequence from the LSTM after 100 epochs of training)

| Actual value | Python | Node | Browser |
|------- | ------ | ---- | ------- |
| 496700.0023 | 498856.75 | 520751.4526 | 523446.835 |
| 165200.0039 | 160055.4375 | 164757.8019 | 165570.8945 |
| 89399.99931 | 107006.8359 | 112949.8694 | 112468.4571 |
| 288099.9948 | 292188.5 | 294070.1243 | 298316.8641 |
| 124899.9997 | 109650.5469 | 116382.986 | 115640.7381 |
| 76799.99851 | 97550.46094 | 99806.35774 | 100445.6921 |
| 331100.0101 | 328920.3438 | 330094.772 | 325816.3978 |
| 117599.9973 | 109937.7109 | 118252.5275 | 116928.3638 |
| 612700.0032 | 613669.3125 | 602983.1445 | 602092.3216 |
| 91700.00133 | 99538.33594 | 103163.5534 | 103240.7722 |
| 895200 | 890565.3125 | 888041.1434 | 887452.5978 |

(They're pretty comparable)

## Run in node
```
npm install
npm start
```

Sample results:
```
Parsing data: 1 ms
Initializing model: 664 ms
Inference: 3584 ms
Training: 3905974 ms
Inference: 4453 ms

[0/100] Average Loss: 0.1258649201534519;
[99/100] Average Loss: 0.0003284691085010394;

Initial predictions:
Prediction | Actual
469420.0671851635  | 496700.00234246254
467031.05384111404 | 165200.00385791063
466522.1437215805  | 89399.99930672348
467918.6033189297  | 288099.994802475
466732.766020298   | 124899.99967962503
466412.33165860176 | 76799.99850802124
468223.6425101757  | 331100.010111928
466688.31949830055 | 117599.99728873372
470178.419983387   | 612700.0032007694
466519.22835707664 | 91700.00132992864
472411.128872633   | 895200

Final predictions:
Prediction | Actual
520751.45261883736 | 496700.00234246254
164757.80188143253 | 165200.00385791063
112949.86943155527 | 89399.99930672348
294070.12426555157 | 288099.994802475
116382.98604860902 | 124899.99967962503
99806.35773986578 | 76799.99850802124
330094.7719722986 | 331100.010111928
118252.52747014165 | 117599.99728873372
602983.1444561481 | 612700.0032007694
103163.55340629816 | 91700.00132992864
888041.1434471607 | 895200
```

## Run in the browser
```
npm install
npm run local
```

Navigate to `localhost:8080`, and press the "Train" button. Timings are logged
to the console.

Sample results:
```
Parsing data: 1 ms
Initializing model: 1445 ms
Inference: 6710 ms
Training: 1382144 ms
Inference: 2924 ms

[0/100] Average Loss: 0.12768243271222507;
[99/100] Average Loss: 0.00038798846885997176;

Initial predictions:
Prediction | Actual
461527.4338454008 | 496700.00234246254
464953.1917244196 | 165200.00385791063
465653.85099351406 | 89399.99930672348
463720.45285999775 | 288099.994802475
465353.2871425152 | 124899.99967962503
465793.40488910675 | 76799.99850802124
463278.57055068016 | 331100.010111928
465421.44018113613 | 117599.99728873372
460372.642621398 | 612700.0032007694
465651.4982432127 | 91700.00132992864
456857.071056962 | 895200

Final predictions:
Prediction | Actual
523446.83496952057 | 496700.00234246254
165570.89448422194 | 165200.00385791063
112468.45708116889 | 89399.99930672348
298316.86413288116 | 288099.994802475
115640.73808193207 | 124899.99967962503
100445.69206088781 | 76799.99850802124
325816.39784276485 | 331100.010111928
116928.36379781365 | 117599.99728873372
602092.3215866089 | 612700.0032007694
103240.7722055912 | 91700.00132992864
887452.5978446007 | 895200
```

## Run in python
Warning: I am not a python developer, so if this isn't right, 🤷‍♀️. First, figure
out how to [install tensorflow](https://www.tensorflow.org/install) since
this is a dependency of the project. It will barf errors if you don't have installed
correctly along the lines of "No module named tensorflow". I don't know how to help with that.

Then:

```
npm run python
# or you can run python3 run.py, it's the same thing
# i'm just bad at python.
```

Sample results:
```
Parsing data: 1 ms
Initializing model: 40 ms
Inference: 2023 ms
Training: 250627 ms
Inference: 1751 ms

[0/100] Average Loss 0.005488885
[99/100] Average Loss 0.000008494

Initial predictions:
451053.062500 | 496700.000000
462004.093750 | 165200.000000
464306.250000 | 89400.000000
457688.687500 | 288100.000000
463628.531250 | 124900.000000
464952.531250 | 76800.000000
456469.250000 | 331100.000000
463662.500000 | 117600.000000
448450.562500 | 612700.000000
464475.281250 | 91700.000000
438155.250000 | 895200.000000

Final predictions:
Prediction | Actual
498856.750000 | 496700.000000
160055.437500 | 165200.000000
107006.835938 | 89400.000000
292188.500000 | 288100.000000
109650.546875 | 124900.000000
97550.460938 | 76800.000000
328920.343750 | 331100.000000
109937.710938 | 117600.000000
613669.312500 | 612700.000000
99538.335938 | 91700.000000
890565.312500 | 895200.000000
```
