'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const cost = require('./lib/cost')
const util = require('./lib/util')
const { getLayerResult } = require('./lib/layer')

;(async function () {
  const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)

  const currentImage = await util.loadImage('./images/louvre.jpg')
  const styleImage = await util.loadImage('./images/monet_800600.jpg')
  const noiseImage = util.generateNoiseImage(currentImage)

  console.log({ currentImage, styleImage })

  const generatedActivation = getLayerResult(vgg19, styleImage, 'block4_conv2')
  const rawActivation = getLayerResult(vgg19, currentImage, 'block4_conv2')

  const contentCost = cost.computeContentCost(rawActivation, generatedActivation)
  const styleCost = cost.computeStyleCost(vgg19, styleImage)

  const totalCost = cost.computeTotalCost(contentCost, styleCost, 10, 40)

  console.log({ contentCost, styleCost, totalCost })

  const optimizer = tf.train.adam(2)
  const trainStep = optimizer.minimize(totalCost)
})(console.error)
