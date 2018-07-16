'use strict'

function getLayerResult (model, input, layerName) {
  let currentResult = input
  let idx = 1

  while (true) {
    const layer = model.getLayer(null, idx++)
    if (layer.name === layerName) return layer.apply(currentResult)
    currentResult = layer.apply(currentResult)
  }
}

module.exports = { getLayerResult }
