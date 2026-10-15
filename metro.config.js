const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Daftarkan ekstensi .tflite sebagai aset
config.resolver.assetExts.push('tflite');

module.exports = config;