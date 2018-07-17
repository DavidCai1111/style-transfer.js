'use strict'
const path = require('path')
const cost = require('./cost')
const util = require('./util')
const { getLayerResult } = require('./layer')

async function run (contentImagePath, styleImagePath, outputImagePath, gpu) {
  let tf
  if (gpu !== undefined) {
    require('@tensorflow/tfjs-node-gpu')
    tf = require('@tensorflow/tfjs')
  } else {
    require('@tensorflow/tfjs-node')
    tf = require('@tensorflow/tfjs')
    tf.setBackend('tensorflow')
  }

  const dir = path.join(__dirname, '..')

  const vgg19 = await tf.loadModel(`file://${dir}/vgg19-tensorflowjs-model/model/model.json`)

  const currentImage = await util.loadImage(contentImagePath)
  const styleImage = await util.loadImage(styleImagePath)

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

  for (let i = 0; i < 2000; i++) {
    const start = Date.now()
    const cost = optimizer.minimize(() => loss(), true, [outputImage])
    console.log(`epoch: ${i + 1}/2000, cost: ${cost.dataSync()}, use ${(Date.now() - start) / 1000}s`)
  }

  util.saveImage(outputImagePath, outputImage)
}

module.exports = { run }
