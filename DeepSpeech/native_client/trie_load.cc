#include <algorithm>
#include <iostream>
#include <string>

#include "ctcdecode/scorer.h"
#include "fst/fstlib.h"
#include "alphabet.h"

using namespace std;


int main(int argc, char** argv)
{
  const char* kenlm_path    = argv[1];
  const char* trie_path     = argv[2];
  const char* alphabet_path = argv[3];

  printf("Loading trie(%s) and alphabet(%s)\n", trie_path, alphabet_path);

  Alphabet alphabet(alphabet_path);
  Scorer scorer(0.0, 0.0, kenlm_path, trie_path, alphabet);

  return 0;
}
