package org.mozilla.deepspeech.libdeepspeech.test;

import android.content.Context;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Test;
import org.junit.FixMethodOrder;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;

import static org.junit.Assert.*;

import org.mozilla.deepspeech.libdeepspeech.DeepSpeechModel;
import org.mozilla.deepspeech.libdeepspeech.Metadata;

import java.io.RandomAccessFile;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class BasicTest {

    public static final String modelFile    = "/data/local/tmp/test/output_graph.tflite";
    public static final String alphabetFile = "/data/local/tmp/test/alphabet.txt";
    public static final String lmFile       = "/data/local/tmp/test/lm.binary";
    public static final String trieFile     = "/data/local/tmp/test/trie";
    public static final String wavFile      = "/data/local/tmp/test/LDC93S1.wav";

    public static final int N_CEP      = 26;
    public static final int N_CONTEXT  = 9;
    public static final int BEAM_WIDTH = 50;

    public static final float LM_ALPHA = 0.75f;
    public static final float LM_BETA  = 1.85f;

    private char readLEChar(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        return (char)((b2 << 8) | b1);
    }

    private int readLEInt(RandomAccessFile f) throws IOException {
        byte b1 = f.readByte();
        byte b2 = f.readByte();
        byte b3 = f.readByte();
        byte b4 = f.readByte();
        return (int)((b1 & 0xFF) | (b2 & 0xFF) << 8 | (b3 & 0xFF) << 16 | (b4 & 0xFF) << 24);
    }

    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getTargetContext();

        assertEquals("org.mozilla.deepspeech.libdeepspeech.test", appContext.getPackageName());
    }

    @Test
    public void loadDeepSpeech_basic() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile, N_CEP, N_CONTEXT, alphabetFile, BEAM_WIDTH);
        m.destroyModel();
    }

    private String metadataToString(Metadata m) {
        String retval = "";
        for (int i = 0; i < m.getNum_items(); ++i) {
            retval += m.getItem(i).getCharacter();
        }
        return retval;
    }

    private String doSTT(DeepSpeechModel m, boolean extendedMetadata) {
        try {
            RandomAccessFile wave = new RandomAccessFile(wavFile, "r");

            wave.seek(20); char audioFormat = this.readLEChar(wave);
            assert (audioFormat == 1); // 1 is PCM

            wave.seek(22); char numChannels = this.readLEChar(wave);
            assert (numChannels == 1); // MONO

            wave.seek(24); int sampleRate = this.readLEInt(wave);
            assert (sampleRate == 16000); // 16000 Hz

            wave.seek(34); char bitsPerSample = this.readLEChar(wave);
            assert (bitsPerSample == 16); // 16 bits per sample

            wave.seek(40); int bufferSize = this.readLEInt(wave);
            assert (bufferSize > 0);

            wave.seek(44);
            byte[] bytes = new byte[bufferSize];
            wave.readFully(bytes);

            short[] shorts = new short[bytes.length/2];
            // to turn bytes to shorts as either big endian or little endian.
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

            if (extendedMetadata) {
                return metadataToString(m.sttWithMetadata(shorts, shorts.length, sampleRate));
            } else {
                return m.stt(shorts, shorts.length, sampleRate);
            }
        } catch (FileNotFoundException ex) {

        } catch (IOException ex) {

        } finally {

        }

        return "";
    }

    @Test
    public void loadDeepSpeech_stt_noLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile, N_CEP, N_CONTEXT, alphabetFile, BEAM_WIDTH);

        String decoded = doSTT(m, false);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.destroyModel();
    }

    @Test
    public void loadDeepSpeech_stt_withLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile, N_CEP, N_CONTEXT, alphabetFile, BEAM_WIDTH);
        m.enableDecoderWihLM(alphabetFile, lmFile, trieFile, LM_ALPHA, LM_BETA);

        String decoded = doSTT(m, false);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.destroyModel();
    }

    @Test
    public void loadDeepSpeech_sttWithMetadata_noLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile, N_CEP, N_CONTEXT, alphabetFile, BEAM_WIDTH);

        String decoded = doSTT(m, true);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.destroyModel();
    }

    @Test
    public void loadDeepSpeech_sttWithMetadata_withLM() {
        DeepSpeechModel m = new DeepSpeechModel(modelFile, N_CEP, N_CONTEXT, alphabetFile, BEAM_WIDTH);
        m.enableDecoderWihLM(alphabetFile, lmFile, trieFile, LM_ALPHA, LM_BETA);

        String decoded = doSTT(m, true);
        assertEquals("she had your dark suit in greasy wash water all year", decoded);
        m.destroyModel();
    }
}
