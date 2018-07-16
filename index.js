'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const cost = require('./lib/cost')
const util = require('./lib/util')
const { getLayerResult } = require('./lib/layer')

tf.setBackend('tensorflow')

;(async function () {
  const vgg19 = await tf.loadModel(`file://${__dirname}/vgg19-tensorflowjs-model/model/model.json`)

  const currentImage = await util.loadImage('./images/louvre.jpg')
  const styleImage = await util.loadImage('./images/monet_800600.jpg')

  const rawActivation = getLayerResult(vgg19, currentImage, 'block4_conv2')
  let outputImage = util.generateNoiseImage(currentImage)

  const loss = () => {
    const contentCost = cost.computeContentCost(
      rawActivation,
      getLayerResult(vgg19, outputImage, 'block4_conv2')
    )

    const styleCost = cost.computeStyleCost(vgg19, styleImage, outputImage)
    const totalCost = cost.computeTotalCost(contentCost, styleCost, 10, 40)

    console.log({
      contentCost: contentCost.dataSync(),
      styleCost: styleCost.dataSync(),
      totalCost: totalCost.dataSync()
    })

    return totalCost
  }

  const optimizer = tf.train.adam(2)

  for (let i = 0; i < 200; i++) {
    const start = Date.now()
    console.log({ outputImage: outputImage.dataSync() })
    const cost = optimizer.minimize(() => loss(), true, [outputImage])
    console.log({ outputImage: outputImage.dataSync() })
    console.log(`cost: ${cost.dataSync()}, use ${(Date.now() - start) / 1000}s`)
  }
})(console.error)
