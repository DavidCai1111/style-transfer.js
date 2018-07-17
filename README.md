# style-transfer.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/style-transfer.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/style-transfer.js)

Generate novel artistic images in Node.js.

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

```sh
node ./transfer.js transfer <contentImagePath> <styleImagePath> <outputImagePath>
// Example: node ./transfer.js transfer ./images/chairs.jpg ./images/Starry_Night.jpg ./out.jpg
```
