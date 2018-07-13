'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const { computeContentCost, computeGramMatrix, computeLayerStyleCost } = require('./cost')

;(async function () {
  const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)
})(console.error)
