### chargan

This package is an attempt to build a useful text-generator using conditional GANs.

The steps are roughly:

- train a character-level autoencoder, producing a `character_encoder` and `character_decoder`
- using a powerful pretrained model, get a vector representation of every string in the dataset
- define a generator that creates vectors that could be passed to the `character_decoder`, conditioned on the vector representation from the pretrained language model
- define a critic that disambiguates between `character_encoder(x) | vec(x)` and `generator(noise, vec(x))`
- use the `character_decoder` to decode the outputs from the `generator`

The process could be useful for text-based data augmentation tasks, like generating more data for NLU or new ways to phrase questions for QA.