'use stirct'
const tf = require('@tensorflow/tfjs')

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

  return contentCost
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

  return layerStyleCost
}

module.exports = {
  computeContentCost,
  computeGramMatrix,
  computeLayerStyleCost
}
