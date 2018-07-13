'use strict'
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

;(async function () {
  const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)

})(console.error)
