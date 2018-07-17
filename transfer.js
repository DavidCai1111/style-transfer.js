'use strict'
const transfer = require('commander')
const pkg = require('./package')
const model = require('./lib/model')

transfer.version(pkg.version)

transfer
  .command('transfer <contentImagePath> <styleImagePath> <outputImagePath>')
  .description('tranfer the style of the "content image"')
  .action(function (contentImagePath, styleImagePath, outputImagePath) {
    ;(async function () {
      await model.run(contentImagePath, styleImagePath, outputImagePath)
    })(console.error)
  })

transfer.parse(process.argv)
