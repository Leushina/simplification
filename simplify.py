import tensorflow as tf
import numpy as np
import torch
import os

from get_model import create_model, preprocess_sentence, preprocess
from get_transformer import create_transformer, translate

# https://www.tensorflow.org/tutorials/text/nmt_with_attention


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_transformer()

    src = "The evaluation was prepared by Human Resource Tactics in North Carolina ."
    simp_src = translate(model, src)
    print(src)
    print(simp_src)
    """
    units = 512
    batch_size = 128
    encoder, decoder, max_length_targ, max_length_inp, inp_lang, targ_lang = create_model(units, batch_size)

    path_to_enc = 'encoder_more_data/encoder'
    path_to_dec = 'decoder_more_data/decoder'
    # encoder.save_weights(path_to_enc)
    # decoder.save_weights(path_to_dec)

    encoder.load_weights(path_to_enc)
    decoder.load_weights(path_to_dec)

    translate('Because the Air Force did not consider her experience as combat ,' +
              'she was prevented from competing for a combat-leadership role .')

    """
