'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const cost = require('./lib/cost')
const util = require('./lib/util')
const { getLayerResult } = require('./lib/layer')

tf.setBackend('tensorflow')

;(async function () {
  const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)

  const currentImage = await util.loadImage('./images/great-wall.jpg')
  const styleImage = await util.loadImage('./images/Starry_Night.jpg')

  const rawActivation = getLayerResult(vgg19, currentImage, 'block4_conv2')
  let outputImage = util.generateNoiseImage(currentImage)

  const loss = () => {
    const contentCost = cost.computeContentCost(
      rawActivation,
      getLayerResult(vgg19, outputImage, 'block4_conv2')
    )

    const styleCost = cost.computeStyleCost(vgg19, styleImage, outputImage)
    const totalCost = cost.computeTotalCost(contentCost, styleCost, 10, 40)

    return totalCost
  }

  const optimizer = tf.train.adam(2)

  for (let i = 0; i < 500; i++) {
    const start = Date.now()
    const cost = optimizer.minimize(() => loss(), true, [outputImage])
    console.log(`epoch: ${i + 1}, cost: ${cost.dataSync()}, use ${(Date.now() - start) / 1000}s`)
  }

  util.saveImage('./out.jpg', outputImage)
})(console.error)
