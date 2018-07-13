'use stirct'
const tf = require('@tensorflow/tfjs')

const STYLE_LAYERS = [
  ['conv1_1', 0.2],
  ['conv2_1', 0.2],
  ['conv3_1', 0.2],
  ['conv4_1', 0.2],
  ['conv5_1', 0.2]
]

function computeContentCost (rawContentActivation, generatedContentActivation) {
  const [, nH, nW, nC] = generatedContentActivation.shape

  const rawContentActivationUnrolled = tf.transpose(rawContentActivation)
  const generatedContentActivationUnrolled = tf.transpose(generatedContentActivation)

  const contentCost = tf.mul(
    (1 / (4 * nH * nW * nC)),
    tf.pow(
      generatedContentActivationUnrolled.sub(rawContentActivationUnrolled), 2
    ).sum()
  )

  return contentCost.asType('float32')
}

function computeGramMatrix (Activation) {
  const GramMatrix = tf.matMul(Activation, tf.transpose(Activation))

  return GramMatrix
}

function computeLayerStyleCost (rawContentActivation, generatedContentActivation) {
  const [, nH, nW, nC] = generatedContentActivation.shape

  rawContentActivation = tf.transpose(
    tf.reshape(rawContentActivation, [nH * nW, nC])
  )
  generatedContentActivation = tf.transpose(
    tf.reshape(generatedContentActivation, [nH * nW, nC])
  )

  const rawContentGramMatrix = computeGramMatrix(rawContentActivation)
  const generatedContentGramMatrix = computeGramMatrix(generatedContentActivation)

  const layerStyleCost = tf.mul(
    (1 / (4 * (nH ** 2) * ((nH * nW) ** 2))),
    tf.pow(
      rawContentGramMatrix.sub(generatedContentGramMatrix), 2
    ).sum()
  )

  return layerStyleCost.asType('float32')
}

function computeStyleCost (model, session) {
  let styleCost = 0

  for (const [layerName, coeff] of STYLE_LAYERS) {
    const out = model[layerName]
    const activation = session.run(out)

    const generatedActivation = out

    const layerCost = computeLayerStyleCost(activation, generatedActivation)

    styleCost += coeff * layerCost
  }

  return styleCost
}

function computeTotalCost (contentCost, styleCost, alpha = 10, beta = 40) {
  return alpha * contentCost + beta * styleCost
}

module.exports = {
  computeContentCost,
  computeGramMatrix,
  computeLayerStyleCost,
  computeStyleCost,
  computeTotalCost
}
