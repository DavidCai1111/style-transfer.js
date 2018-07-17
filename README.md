# style-transfer.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/style-transfer.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/style-transfer.js)

Generate novel artistic images in Node.js

## How to use

### Clone this repository

```sh
git clone https://github.com/DavidCai1993/style-transfer.js.git
```

### Download [model](https://github.com/DavidCai1993/vgg19-tensorflowjs-model)

```sh
npm run model
```

### Install dependences

```sh
npm install
```

### Start transfering

```sh
node ./transfer.js transfer <contentImagePath> <styleImagePath> <outputImagePath>
```
