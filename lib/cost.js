'use stirct'
const tf = require('@tensorflow/tfjs')
const { getLayerResult } = require('./layer')

const STYLE_LAYERS = [
  ['block1_conv1', 0.2],
  ['block2_conv1', 0.2],
  ['block3_conv1', 0.2],
  ['block4_conv1', 0.2],
  ['block5_conv1', 0.2]
]

function computeContentCost (rawContentActivation, generatedContentActivation) {
  const [, nH, nW, nC] = generatedContentActivation.shape

  const rawContentActivationUnrolled = tf.transpose(rawContentActivation).reshape([nC, nH * nW])
  const generatedContentActivationUnrolled = tf.transpose(generatedContentActivation).reshape([nC, nH * nW])

  const contentCost = tf.mul(
    (1 / (4 * nH * nW * nC)),
    tf.square(
      generatedContentActivationUnrolled.sub(rawContentActivationUnrolled)
    ).sum()
  )

  return contentCost
}

function computeGramMatrix (activation) {
  const GramMatrix = tf.matMul(activation, tf.transpose(activation))

  return GramMatrix
}

function computeLayerStyleCost (rawContentActivation, generatedContentActivation) {
  const [, nH, nW, nC] = generatedContentActivation.shape

  rawContentActivation = tf.transpose(rawContentActivation)
    .reshape([nC, nH * nW])
  generatedContentActivation = tf.transpose(generatedContentActivation)
    .reshape([nC, nH * nW])

  const rawContentGramMatrix = computeGramMatrix(rawContentActivation)
  const generatedContentGramMatrix = computeGramMatrix(generatedContentActivation)

  const layerStyleCost = tf.mul(
    (1 / (4 * nC * nC * nH * nH * nW * nW)),
    tf.square(
      rawContentGramMatrix.sub(generatedContentGramMatrix)
    ).sum()
  )

  return layerStyleCost
}

function computeStyleCost (model, inputImage, generatedImage) {
  let styleCost = tf.scalar(0)

  for (const [layerName, coeff] of STYLE_LAYERS) {
    const activation = getLayerResult(model, inputImage, layerName)
    const generatedActivation = getLayerResult(model, generatedImage, layerName)

    const layerCost = computeLayerStyleCost(activation, generatedActivation)

    styleCost = styleCost.add(tf.scalar(coeff).mul(layerCost))
  }

  return styleCost
}

function computeTotalCost (contentCost, styleCost, alpha = 10, beta = 40) {
  return tf.scalar(alpha).mul(contentCost).add(tf.scalar(beta).mul(styleCost))
}

module.exports = {
  computeContentCost,
  computeGramMatrix,
  computeLayerStyleCost,
  computeStyleCost,
  computeTotalCost
}
