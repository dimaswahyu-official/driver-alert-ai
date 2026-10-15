import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';
import { useResizer } from 'react-native-vision-camera-resizer';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';

// --- KALKULASI EAR ---
const calculateEAR = (landmarks: Float32Array | number[], indices: number[]): number => {
  'worklet';

  const getPoint = (idx: number) => ({
    x: landmarks[idx * 3],
    y: landmarks[idx * 3 + 1],
  });

  const p1 = getPoint(indices[0]);
  const p2 = getPoint(indices[1]);
  const p3 = getPoint(indices[2]);
  const p4 = getPoint(indices[3]);
  const p5 = getPoint(indices[4]);
  const p6 = getPoint(indices[5]);

  const dist = (a: { x: number, y: number }, b: { x: number, y: number }) =>
    Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));

  const vertical1 = dist(p2, p6);
  const vertical2 = dist(p3, p5);
  const horizontal = dist(p1, p4);

  if (horizontal === 0) return 0;
  return (vertical1 + vertical2) / (2.0 * horizontal);
};

export default function RootLayout() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');

  const [isDrowsy, setIsDrowsy] = useState(false);

  const model = useTensorflowModel(
    require('../../assets/face_landmark.tflite'),
    [],
  );

  const { resizer } = useResizer({
    width: 192,
    height: 192,
    channelOrder: 'rgb',
    dataType: 'float32',
    pixelLayout: 'interleaved',
    scaleMode: 'cover',
  });

  const handleAIResult = (drowsyStatus: boolean) => {
    setIsDrowsy(drowsyStatus);
  };

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const isProcessing = useSharedValue(false);
  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    onFrame(frame) {
      'worklet';
      if (isProcessing.value) {
        if (frame.isValid) frame.dispose();
        return;
      }
      try {
        const tflite = model.model;
        const frameResizer = resizer;

        if (
          model.state !== 'loaded' ||
          tflite == null ||
          frameResizer == null
        ) {
          return;
        }

        const resized = frameResizer.resize(frame);

        try {
          isProcessing.value = true;
          const inputBuffer = resized.getPixelBuffer();
          const outputs = tflite.runSync([inputBuffer]);

          if (!outputs || outputs.length === 0 || outputs[0] == null) {
            return;
          }

          const landmarks = new Float32Array(outputs[0]);

          if (landmarks.length < 1404) {
            return;
          }

          const LEFT_EYE = [33, 160, 158, 133, 153, 144];
          const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

          const leftEAR = calculateEAR(landmarks, LEFT_EYE);
          const rightEAR = calculateEAR(landmarks, RIGHT_EYE);
          const averageEAR = (leftEAR + rightEAR) / 2;

          scheduleOnRN(() => {
            handleAIResult(averageEAR < 0.20);
          });
        } finally {
          isProcessing.value = false;
          if (frame.isValid) {
            frame.dispose();
          }
        }
      } finally {
        if (frame.isValid) {
          frame.dispose();
        }
      }
    },
  });

  if (!hasPermission) {
    return <Text style={styles.centerText}>Need camera access</Text>;
  }

  if (!device) {
    return <Text style={styles.centerText}>No front camera found</Text>;
  }

  if (model.state === 'error') {
    return <Text style={styles.centerText}>Failed to load AI model</Text>;
  }

  if (model.state !== 'loaded' || !model.model) {
    return <Text style={styles.centerText}>Loading AI Model...</Text>;
  }

  return (
    <View style={StyleSheet.absoluteFill}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        // outputs={[frameOutput]}
      />

      <View style={styles.overlay}>
        <Text
          style={[
            styles.statusText,
            isDrowsy ? styles.danger : styles.safe,
          ]}
        >
          {isDrowsy ? '⚠️ TERTIDUR!' : 'Mata Terbuka'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  centerText: {
    flex: 1,
    textAlign: 'center',
    textAlignVertical: 'center',
    fontSize: 18,
    fontWeight: 'bold'
  },
  overlay: {
    position: 'absolute',
    bottom: 50,
    width: '100%',
    alignItems: 'center',
  },
  statusText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  safe: {
    backgroundColor: 'rgba(16, 185, 129, 0.9)',
  },
  danger: {
    backgroundColor: 'rgba(239, 68, 68, 0.9)',
  }
});