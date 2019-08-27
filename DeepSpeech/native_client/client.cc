#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <sstream>
#include <string>

#if defined(__ANDROID__) || defined(_MSC_VER)
#define NO_SOX
#endif

#if defined(_MSC_VER)
#define NO_DIR
#endif

#ifndef NO_SOX
#include <sox.h>
#endif

#ifndef NO_DIR
#include <dirent.h>
#include <unistd.h>
#endif // NO_DIR
#include <vector>

#include "deepspeech.h"
#include "args.h"

#define N_CEP 26
#define N_CONTEXT 9
#define BEAM_WIDTH 500
#define LM_ALPHA 0.75f
#define LM_BETA 1.85f

typedef struct {
  const char* string;
  double cpu_time_overall;
} ds_result;

struct meta_word {
  std::string word;
  float start_time;
  float duration;
};

char* metadataToString(Metadata* metadata);
std::vector<meta_word> WordsFromMetadata(Metadata* metadata);
char* JSONOutput(Metadata* metadata);

ds_result
LocalDsSTT(ModelState* aCtx, const short* aBuffer, size_t aBufferSize,
           int aSampleRate, bool extended_output, bool json_output)
{
  ds_result res = {0};

  clock_t ds_start_time = clock();

  if (extended_output) {
    Metadata *metadata = DS_SpeechToTextWithMetadata(aCtx, aBuffer, aBufferSize, aSampleRate);
    res.string = metadataToString(metadata);
    DS_FreeMetadata(metadata);
  } else if (json_output) {
    Metadata *metadata = DS_SpeechToTextWithMetadata(aCtx, aBuffer, aBufferSize, aSampleRate);
    res.string = JSONOutput(metadata);
    DS_FreeMetadata(metadata);
  } else if (stream_size > 0) {
    StreamingState* ctx;
    int status = DS_SetupStream(aCtx, 0, aSampleRate, &ctx);
    if (status != DS_ERR_OK) {
      res.string = strdup("");
      return res;
    }
    size_t off = 0;
    const char *last = nullptr;
    while (off < aBufferSize) {
      size_t cur = aBufferSize - off > stream_size ? stream_size : aBufferSize - off;
      DS_FeedAudioContent(ctx, aBuffer + off, cur);
      off += cur;
      const char* partial = DS_IntermediateDecode(ctx);
      if (last == nullptr || strcmp(last, partial)) {
        printf("%s\n", partial);
        last = partial;
      } else {
        DS_FreeString((char *) partial);
      }
    }
    if (last != nullptr) {
      DS_FreeString((char *) last);
    }
    res.string = DS_FinishStream(ctx);
  } else {
    res.string = DS_SpeechToText(aCtx, aBuffer, aBufferSize, aSampleRate);
  }

  clock_t ds_end_infer = clock();

  res.cpu_time_overall =
    ((double) (ds_end_infer - ds_start_time)) / CLOCKS_PER_SEC;

  return res;
}

typedef struct {
  char*  buffer;
  size_t buffer_size;
  int    sample_rate;
} ds_audio_buffer;

ds_audio_buffer
GetAudioBuffer(const char* path)
{
  ds_audio_buffer res = {0};

#ifndef NO_SOX
  sox_format_t* input = sox_open_read(path, NULL, NULL, NULL);
  assert(input);

  // Resample/reformat the audio so we can pass it through the MFCC functions
  sox_signalinfo_t target_signal = {
      16000, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
  };

  sox_signalinfo_t interm_signal;

  sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_SIGN2, // Sample format
    16, // Bits per sample
    0.0, // Compression factor
    sox_option_default, // Should bytes be reversed
    sox_option_default, // Should nibbles be reversed
    sox_option_default, // Should bits be reversed (pairs of bits?)
    sox_false // Reverse endianness
  };

#ifdef __APPLE__
  // It would be preferable to use sox_open_memstream_write here, but OS-X
  // doesn't support POSIX 2008, which it requires. See Issue #461.
  // Instead, we write to a temporary file.
  char* output_name = tmpnam(NULL);
  assert(output_name);
  sox_format_t* output = sox_open_write(output_name, &target_signal,
                                        &target_encoding, "raw", NULL, NULL);
#else
  sox_format_t* output = sox_open_memstream_write(&res.buffer,
                                                  &res.buffer_size,
                                                  &target_signal,
                                                  &target_encoding,
                                                  "raw", NULL);
#endif

  assert(output);

  res.sample_rate = (int)output->signal.rate;

  if ((int)input->signal.rate < 16000) {
    fprintf(stderr, "Warning: original sample rate (%d) is lower than 16kHz. Up-sampling might produce erratic speech recognition.\n", (int)input->signal.rate);
  }

  // Setup the effects chain to decode/resample
  char* sox_args[10];
  sox_effects_chain_t* chain =
    sox_create_effects_chain(&input->encoding, &output->encoding);

  interm_signal = input->signal;

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  sox_args[0] = (char*)input;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &input->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("rate"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("channels"));
  assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  e = sox_create_effect(sox_find_effect("output"));
  sox_args[0] = (char*)output;
  assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &output->signal) ==
         SOX_SUCCESS);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, NULL, NULL);
  sox_delete_effects_chain(chain);

  // Close sox handles
  sox_close(output);
  sox_close(input);
#endif // NO_SOX

#ifdef NO_SOX
  // FIXME: Hack and support only 16kHz mono 16-bits PCM
  FILE* wave = fopen(path, "r");

  size_t rv;

  unsigned short audio_format;
  fseek(wave, 20, SEEK_SET); rv = fread(&audio_format, 2, 1, wave);

  unsigned short num_channels;
  fseek(wave, 22, SEEK_SET); rv = fread(&num_channels, 2, 1, wave);

  unsigned int sample_rate;
  fseek(wave, 24, SEEK_SET); rv = fread(&sample_rate, 4, 1, wave);

  unsigned short bits_per_sample;
  fseek(wave, 34, SEEK_SET); rv = fread(&bits_per_sample, 2, 1, wave);

  assert(audio_format == 1); // 1 is PCM
  assert(num_channels == 1); // MONO
  assert(sample_rate == 16000); // 16000 Hz
  assert(bits_per_sample == 16); // 16 bits per sample

  fprintf(stderr, "audio_format=%d\n", audio_format);
  fprintf(stderr, "num_channels=%d\n", num_channels);
  fprintf(stderr, "sample_rate=%d\n", sample_rate);
  fprintf(stderr, "bits_per_sample=%d\n", bits_per_sample);

  fseek(wave, 40, SEEK_SET); rv = fread(&res.buffer_size, 4, 1, wave);
  fprintf(stderr, "res.buffer_size=%ld\n", res.buffer_size);

  fseek(wave, 44, SEEK_SET);
  res.buffer = (char*)malloc(sizeof(char) * res.buffer_size);
  rv = fread(res.buffer, sizeof(char), res.buffer_size, wave);

  fclose(wave);
#endif // NO_SOX

#ifdef __APPLE__
  res.buffer_size = (size_t)(output->olength * 2);
  res.buffer = (char*)malloc(sizeof(char) * res.buffer_size);
  FILE* output_file = fopen(output_name, "rb");
  assert(fread(res.buffer, sizeof(char), res.buffer_size, output_file) == res.buffer_size);
  fclose(output_file);
  unlink(output_name);
#endif

  return res;
}

void
ProcessFile(ModelState* context, const char* path, bool show_times)
{
  ds_audio_buffer audio = GetAudioBuffer(path);

  // Pass audio to DeepSpeech
  // We take half of buffer_size because buffer is a char* while
  // LocalDsSTT() expected a short*
  ds_result result = LocalDsSTT(context,
                                (const short*)audio.buffer,
                                audio.buffer_size / 2,
                                audio.sample_rate,
                                extended_metadata,
                                json_output);
  free(audio.buffer);

  if (result.string) {
    printf("%s\n", result.string);
    DS_FreeString((char*)result.string);
  }

  if (show_times) {
    printf("cpu_time_overall=%.05f\n",
           result.cpu_time_overall);
  }
}

char*
metadataToString(Metadata* metadata)
{
  std::string retval = "";
  for (int i = 0; i < metadata->num_items; i++) {
    MetadataItem item = metadata->items[i];
    retval += item.character;
  }
  return strdup(retval.c_str());
}

std::vector<meta_word>
WordsFromMetadata(Metadata* metadata)
{
  std::vector<meta_word> word_list;

  std::string word = "";
  float word_start_time = 0;

  // Loop through each character
  for (int i = 0; i < metadata->num_items; i++) {
    MetadataItem item = metadata->items[i];

    // Append character to word if it's not a space
    if (strcmp(item.character, " ") != 0 
        && strcmp(item.character, u8"　") != 0) {
      word.append(item.character);
    }

    // Word boundary is either a space or the last character in the array
    if (strcmp(item.character, " ") == 0
        || strcmp(item.character, u8" ") == 0
        || i == metadata->num_items-1) {

      float word_duration = item.start_time - word_start_time;

      if (word_duration < 0) {
        word_duration = 0;
      }

      meta_word w;
      w.word = word;
      w.start_time = word_start_time;
      w.duration = word_duration;

      word_list.push_back(w);

      // Reset
      word = "";
      word_start_time = 0;
    } else {
      if (word.length() == 1) {
        word_start_time = item.start_time; // Log the start time of the new word
      }
    }
  }

  return word_list;
}

char* 
JSONOutput(Metadata* metadata)
{
  std::vector<meta_word> words = WordsFromMetadata(metadata);

  std::ostringstream out_string;
  out_string << R"({"metadata":{"confidence":)" << metadata->probability << R"(},"words":[)";

  for (int i = 0; i < words.size(); i++) {
    meta_word w = words[i];
    out_string << R"({"word":")" << w.word << R"(","time":)" << w.start_time << R"(,"duration":)" << w.duration << "}";

    if (i < words.size() - 1) {
      out_string << ",";
    }
  }
  
  out_string << "]}\n";

  return strdup(out_string.str().c_str());
}

int
main(int argc, char **argv)
{
  if (!ProcessArgs(argc, argv)) {
    return 1;
  }

  // Initialise DeepSpeech
  ModelState* ctx;
  int status = DS_CreateModel(model, N_CEP, N_CONTEXT, alphabet, BEAM_WIDTH, &ctx);
  if (status != 0) {
    fprintf(stderr, "Could not create model.\n");
    return 1;
  }

  if (lm && (trie || load_without_trie)) {
    int status = DS_EnableDecoderWithLM(ctx,
                                        alphabet,
                                        lm,
                                        trie,
                                        LM_ALPHA,
                                        LM_BETA);
    if (status != 0) {
      fprintf(stderr, "Could not enable CTC decoder with LM.\n");
      return 1;
    }
  }

#ifndef NO_SOX
  // Initialise SOX
  assert(sox_init() == SOX_SUCCESS);
#endif

  struct stat wav_info;
  if (0 != stat(audio, &wav_info)) {
    printf("Error on stat: %d\n", errno);
  }

  switch (wav_info.st_mode & S_IFMT) {
#ifndef _MSC_VER
    case S_IFLNK:
#endif
    case S_IFREG:
        ProcessFile(ctx, audio, show_times);
      break;

#ifndef NO_DIR
    case S_IFDIR:
        {
          printf("Running on directory %s\n", audio);
          DIR* wav_dir = opendir(audio);
          assert(wav_dir);

          struct dirent* entry;
          while ((entry = readdir(wav_dir)) != NULL) {
            std::string fname = std::string(entry->d_name);
            if (fname.find(".wav") == std::string::npos) {
              continue;
            }

            std::ostringstream fullpath;
            fullpath << audio << "/" << fname;
            std::string path = fullpath.str();
            printf("> %s\n", path.c_str());
            ProcessFile(ctx, path.c_str(), show_times);
          }
          closedir(wav_dir);
        }
      break;
#endif

    default:
        printf("Unexpected type for %s: %d\n", audio, (wav_info.st_mode & S_IFMT));
      break;
  }

#ifndef NO_SOX
  // Deinitialise and quit
  sox_quit();
#endif // NO_SOX

  DS_DestroyModel(ctx);

  return 0;
}
