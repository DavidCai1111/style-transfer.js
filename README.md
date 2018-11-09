# style-transfer.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/style-transfer.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/style-transfer.js)

Generate novel artistic images in Node.js, using [TensorFlow.js](https://js.tensorflow.org/).

## Some Examples

![style-transfer.png](http://static.cnodejs.org/FuGdXW7RE0zc4K_vR5OKflSsrejp)

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
node ./transfer.js transfer -c <contentImagePath> -s <styleImagePath> -o <outputImagePath> [--gpu]

// Example: node ./transfer.js transfer -c ./images/louvre.jpg -s ./images/Francis_Picabia.jpg -o output.jpg --gpu
```
