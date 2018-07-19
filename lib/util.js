'use strict'
const tf = require('@tensorflow/tfjs')
const Jimp = require('jimp')

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])

async function loadImage (path) {
  let img = await Jimp.read(path)
  img.resize(400, 300)

  const p = []

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
    p.push(this.bitmap.data[idx + 0])
    p.push(this.bitmap.data[idx + 1])
    p.push(this.bitmap.data[idx + 2])
  })

  return tf.tensor3d(p, [400, 300, 3]).reshape([1, 400, 300, 3]).sub(MEANS)
}

function generateNoiseImage (image, noiseRatio = 0.6) {
  const noiseImage = tf.randomUniform([1, 400, 300, 3], -20, 20)

  return noiseImage.mul(noiseRatio).add(image.mul(1 - noiseRatio)).variable()
}

function saveImage (path, tensor) {
  let newTensor = tensor.add(MEANS).reshape([400, 300, 3])
  const newTensorArray = Array.from(newTensor.dataSync())
  let i = 0

  return new Promise(function (resolve, reject) {
    // eslint-disable-next-line no-new
    new Jimp(400, 300, function (err, image) {
      if (err) return reject(err)
      image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
        this.bitmap.data[idx + 0] = newTensorArray[i++]
        this.bitmap.data[idx + 1] = newTensorArray[i++]
        this.bitmap.data[idx + 2] = newTensorArray[i++]
        this.bitmap.data[idx + 3] = 255
      })

      image.write(path)
      return resolve(null)
    })
  })
}

module.exports = { loadImage, generateNoiseImage, saveImage }
