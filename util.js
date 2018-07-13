'use strict'
const tf = require('@tensorflow/tfjs')
const jimp = require('jimp')

const MEANS = tf.tensor1d([123.68, 116.779, 103.939]).reshape([1, 1, 1, 3])

async function loadImage (path) {
  const img = await jimp.read(path)
  img.resize(224, 224)

  let r = []
  let g = []
  let b = []

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
    r.push(this.bitmap.data[idx + 0])
    g.push(this.bitmap.data[idx + 1])
    b.push(this.bitmap.data[idx + 2])
  })

  return tf.tensor3d(r.concat(g).concat(b), [224, 224, 3])
    .reshape([1, 224, 224, 3])
    .sub(MEANS)
}

module.exports = { loadImage }
