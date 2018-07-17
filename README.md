# style-transfer.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/style-transfer.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/style-transfer.js)

Generate novel artistic images in Node.js.

## Some Examples

![example1.png](http://dn-cnode.qbox.me/Fr5rcQ0-dZGElXl9NBh3Q99cwUdw)

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
node ./transfer.js transfer -c <contentImagePath> -s <styleImagePath> -o o<outputImagePath> [--gpu]

// Example: node ./transfer.js transfer -c ./images/chairs.jpg -s ./images/monet_800600.jpg -o output.jpg --gpu
```
