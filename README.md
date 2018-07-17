# style-transfer.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/style-transfer.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/style-transfer.js)

Generate novel artistic images in Node.js, using [TensorFlow.js](https://js.tensorflow.org/).

## Some Examples

![example1.png](http://dn-cnode.qbox.me/Fn0CW_qNPdqyCNtmvV_2YbnYrm8a)
![example2.png](http://dn-cnode.qbox.me/Fp-YmDHQ794V2NT-hFOVJjM-SCWH)

## How To Use It

### Clone This Repository

```sh
git clone https://github.com/DavidCai1993/style-transfer.js.git

cd ./style-transfer.js
```

### Download The [VGG-19 Model](https://github.com/DavidCai1993/vgg19-tensorflowjs-model)

```sh
npm run model
```

### Install Dependences

```sh
npm install
```

### Start Making Some Great Art

```js
node ./transfer.js transfer -c <contentImagePath> -s <styleImagePath> -o <outputImagePath>

// Example: node ./transfer.js transfer -c ./images/chairs.jpg -s ./images/monet_800600.jpg -o output.jpg
```
