'use strict'
const tf = require('@tensorflow/tfjs')
const cv = require('opencv4nodejs')

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])

async function loadImage (path) {
  let mat = cv.imread(path)
  mat = mat.cvtColor(cv.COLOR_BGR2RGB)
  mat = mat.resize(400, 300)

  return tf.tensor3d(mat.getDataAsArray()).reshape([1, 400, 300, 3]).sub(MEANS)
}

function generateNoiseImage (image, noiseRatio = 0.6) {
  const noiseImage = tf.randomUniform([1, 400, 300, 3], -20, 20)

  return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio)).variable()
}

function saveImage (path, tensor) {
  tensor = tensor.reshape([400, 300, 3])

  const matFromArray = new cv.Mat(
    Buffer.from(Array.from(tensor.dataSync())), 400, 300,
    cv.CV_8UC3
  )

  cv.imwrite(path, matFromArray)
}

module.exports = { loadImage, generateNoiseImage, saveImage }
