'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const { computeContentCost } = require('./cost')

;(async function () {
  // const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)

  const aC = tf.randomNormal([1, 4, 4, 3], 1, 4)
  const aG = tf.randomNormal([1, 4, 4, 3], 1, 4)

  aC.print()
  console.log(computeContentCost(aC, aG).print())
})(console.error)
