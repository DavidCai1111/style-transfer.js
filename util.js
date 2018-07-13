'use strict'
const tf = require('@tensorflow/tfjs')
const jimp = require('jimp')

function loadImage (path) {
  const img = jimp.read(path).resize(224, 224)

  let r = []
  let g = []
  let b = []

  img.scan(0, 0, img.bitmap.width, img.bitmap.height, function (x, y, idx) {
    r.push(this.bitmap.data[idx + 0])
    g.push(this.bitmap.data[idx + 1])
    b.push(this.bitmap.data[idx + 2])
  })

  return tf.tensor3d(r.concat(g).concat(b), [224, 224, 3])
}

module.exports = { loadImage }
