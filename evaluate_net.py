import os

from config import cfg
from lang_dict.lang import LanguageIndex
from net.net import *
from utils.img_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_lang = LanguageIndex()
vocab_size = len(label_lang.word2idx)

BATCH_SIZE = 1
embedding_dim = cfg.EMBEDDING_DIM
units = cfg.UNITS

encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir , checkpoint_name='ckpt', max_to_keep=5)
status=checkpoint.restore(manager.latest_checkpoint)


def evaluate(encoder, decoder, img_path, label_lang):
    img = process_img(img_path)

    enc_output, enc_hidden = encoder(np.expand_dims(img, axis=0))

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([label_lang.word2idx['<start>']] * BATCH_SIZE, 1)

    results = np.zeros((BATCH_SIZE, 25), np.int32)

    for t in range(1, 25):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        predicted_id = tf.argmax(predictions, axis=-1).numpy()

        results[:, t - 1] = predicted_id

        dec_input = tf.expand_dims(predicted_id, 1)

    preds = [process_result(result, label_lang) for result in results]

    print("pred :" + preds[0])


# img_path = "./sample/1_bridleway_9530.jpg"
img_path=r"E:\tsl_file\python_project\CRNN-ATTENTION\example\images\1_E55A.png"

evaluate(encoder=encoder, decoder=decoder, img_path=img_path, label_lang=label_lang)
