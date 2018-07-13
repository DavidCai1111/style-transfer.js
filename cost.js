'use stirct'
const tf = require('@tensorflow/tfjs')

function computeContentCost (rawContentActivation, generatedContentActivation) {
  const [, nH, nW, nC] = generatedContentActivation.shape

  const rawContentActivationUnrolled = tf.transpose(rawContentActivation)
  const generatedContentActivationUnrolled = tf.transpose(generatedContentActivation)

  const contentCost = tf.mul(
    (1 / (4 * nH * nW * nC)),
    tf.pow((generatedContentActivationUnrolled.sub(rawContentActivationUnrolled)), 2).sum()
  )

  return contentCost
}

module.exports = { computeContentCost }
