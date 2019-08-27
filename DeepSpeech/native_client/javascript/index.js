'use strict';

const binary = require('node-pre-gyp');
const path = require('path')
// 'lib', 'binding', 'v0.1.1', ['node', 'v' + process.versions.modules, process.platform, process.arch].join('-'), 'deepspeech-bingings.node')
const binding_path = binary.find(path.resolve(path.join(__dirname, 'package.json')));

// On Windows, we can't rely on RPATH being set to $ORIGIN/../ or on
// @loader_path/../ but we can change the PATH to include the proper directory
// for the dynamic linker
if (process.platform === 'win32') {
  const dslib_path = path.resolve(path.join(binding_path, '../..'));
  var oldPath = process.env.PATH;
  process.env['PATH'] = `${dslib_path};${process.env.PATH}`;
}

const binding = require(binding_path);

if (process.platform === 'win32') {
  process.env['PATH'] = oldPath;
}

function Model() {
    this._impl = null;

    const rets = binding.CreateModel.apply(null, arguments);
    const status = rets[0];
    const impl = rets[1];
    if (status !== 0) {
        throw "CreateModel failed with error code " + status;
    }

    this._impl = impl;
}

Model.prototype.enableDecoderWithLM = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    return binding.EnableDecoderWithLM.apply(null, args);
}

Model.prototype.stt = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    return binding.SpeechToText.apply(null, args);
}

Model.prototype.sttWithMetadata = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    return binding.SpeechToTextWithMetadata.apply(null, args);
}

Model.prototype.setupStream = function() {
    const args = [this._impl].concat(Array.prototype.slice.call(arguments));
    const rets = binding.SetupStream.apply(null, args);
    const status = rets[0];
    const ctx = rets[1];
    if (status !== 0) {
        throw "SetupStream failed with error code " + status;
    }
    return ctx;
}

Model.prototype.feedAudioContent = function() {
    binding.FeedAudioContent.apply(null, arguments);
}

Model.prototype.intermediateDecode = function() {
    return binding.IntermediateDecode.apply(null, arguments);
}

Model.prototype.finishStream = function() {
    return binding.FinishStream.apply(null, arguments);
}

Model.prototype.finishStreamWithMetadata = function() {
    return binding.FinishStreamWithMetadata.apply(null, arguments);
}

function DestroyModel(model) {
    return binding.DestroyModel(model._impl);
}

module.exports = {
    Model: Model,
    printVersions: binding.PrintVersions,
    DestroyModel: DestroyModel,
    FreeMetadata: binding.FreeMetadata
};
