import os
import os.path as osp
import time

from tqdm import *

from config import cfg
from lang_dict.lang import LanguageIndex
from net.net_new import *
import math

from utils.img_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
类别翻译模型，数据decoder需要开始和结束的标志，这里可能也需要类似的技巧
'''

def max_length(tensor):
    return max(len(t) for t in tensor)


root = "example"
root=r'E:/tsl_file/python_project/all_datas'


def create_dataset_from_dir(root):
    img_names = os.listdir(root)
    img_paths = []
    for img_name in tqdm(img_names, desc="read dir:"):
        img_name = img_name.rstrip().strip()
        img_path = root + "/" + img_name
        # print('path exist',os.path.exists(img_path))
        if osp.exists(img_path):
            img_paths.append(img_path)
    labels = [img_path.split("/")[-1].split("_")[-2] for img_path in tqdm(img_paths, desc="generator label:")]
    # print('lables={},\nimg_pahts={}'.format(labels,img_path))
    return img_paths, labels


def create_dataset_from_file(root, file_path):
    with open(osp.join(root, file_path), "r") as f:
        readlines = f.readlines()

    img_paths = []
    for img_name in tqdm(readlines, desc="read dir:"):
        img_name = img_name.rstrip().strip()
        img_name = img_name.split(" ")[0]
        img_path = root + "/" + img_name
        # if osp.exists(img_path):
        img_paths.append(img_path)
    img_paths = img_paths[:1000000]
    labels = [img_path.split("/")[-1].split("_")[-1] for img_path in tqdm(img_paths, desc="generator label:")]
    labels=[each.split('.')[0] for each in labels]
    # print('labels_new={},\n,img_paths={}'.format(labels,img_paths))
    return img_paths, labels


def load_dataset(root):
    img_paths_tensor, labels = create_dataset_from_file(root, "annotation_attention.txt")

    labels = [label for label in labels]

    processed_labels = [preprocess_label(label) for label in tqdm(labels, desc="process label:")]
    # print('processed_labels ',processed_labels )

    label_lang = LanguageIndex()

    labels_tensor = [[label_lang.word2idx[s] for s in label.split(' ')] for label in processed_labels]

    label_max_len = max_length(labels_tensor)

    labels_tensor = tf.keras.preprocessing.sequence.pad_sequences(labels_tensor, maxlen=label_max_len, padding='post')

    return img_paths_tensor, labels_tensor, labels, label_lang, label_max_len


img_paths_tensor, labels_tensor, labels, label_lang, label_max_len = load_dataset(root)

BATCH_SIZE = cfg.TRAIN_BATCH_SIZE
N_BATCH = len(img_paths_tensor) // BATCH_SIZE
embedding_dim = cfg.EMBEDDING_DIM
units = cfg.UNITS

vocab_size = len(label_lang.word2idx)


def map_func(img_path_tensor, label_tensor, label):
    # print('img_path_tensor',img_path_tensor)

    image = tf.io.read_file(img_path_tensor)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    imread = tf.image.resize(image, (32, 100))

    # imread = cv2.imread(img_path_tensor, cv2.IMREAD_GRAYSCALE)
    # if imread is None:
    #     print(img_path_tensor)
    # imread = resize_image(imread, 100, 32)
    # print('imread.shape',imread.shape)
    # imread = np.expand_dims(imread, axis=-1)#最后一维看情况需不需要扩充
    imread = np.array(imread, np.float32)
    # print('labels',label_tensor, label)
    return imread, label_tensor, label

# def _decode_and_resize( filename, label):
#         image = tf.io.read_file(filename)
#         image = tf.io.decode_jpeg(image, channels=1)
#         image = tf.image.convert_image_dtype(image, tf.float32)
#         image = tf.image.resize(image, (32, image_width))
#         return image, label


dataset = tf.data.Dataset.from_tensor_slices((img_paths_tensor, labels_tensor, labels)) \
    .map(lambda item1, item2, item3: tf.py_function(map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.string]),
         num_parallel_calls=2) \
    .shuffle(10000, reshuffle_each_iteration=True).prefetch(2)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(units, BATCH_SIZE)
# encoder = Encoder()
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

# global_step = tf.train.get_or_create_global_step()
# global_step=tf.train.create_global_step()
start_learning_rate = cfg.LEARNING_RATE
learning_rate = tf.Variable(start_learning_rate, dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

#
# checkpoint_dir = './checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
#
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



checkpoint_dir = './checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir , checkpoint_name='ckpt', max_to_keep=5)

status=checkpoint.restore(manager.latest_checkpoint)
print('status',status)
print('optimizer.iterations.numpy()',optimizer.iterations.numpy())#optimizer.iterations.numpy() 1062
lr = max(0.00001, start_learning_rate  * math.pow(0.99, optimizer.iterations.numpy()//30))
learning_rate.assign(lr)




EPOCHS = 200

logdir = "./logs/"
writer = tf.summary.create_file_writer(logdir)
# writer.set_as_default()
# print('dataset',dataset)

# for (batch, (inp, targ, ground_truths)) in enumerate(dataset):
#     print('batch', batch)
#     print('inp shape', inp.shape)  # inp shape (64, 32, 100, 1, 1)




with writer.as_default():
# if True:
    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0
        lr = max(0.00001, start_learning_rate * math.pow(0.99, epoch))
        learning_rate.assign(lr)
        # print('start')

        for (batch, (inp, targ, ground_truths)) in enumerate(dataset):
            # print('batch',batch)
            # print('inp shape',inp.shape)#inp shape (64, 32, 100, 1, 1)
            # step=epoch*N_BATCH+batch
            # print('epoch={},step={},batch={}'.format(epoch,step,batch))
            loss = 0
            # global_step.assign_add(1)

            results = np.zeros((BATCH_SIZE, targ.shape[1] - 1), np.int32)

            with tf.GradientTape() as tape:
                enc_output ,enc_hidden= encoder(inp)
                #24是自定义pooling层获取
                # print('enc_output shape',enc_output.shape)#shape=(30, 24, 512)=(batch_size,)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([label_lang.word2idx['<start>']] * BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                    predicted_id = tf.argmax(predictions, axis=-1).numpy()

                    results[:, t - 1] = predicted_id

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            total_loss += batch_loss

            # variables = encoder.variables + decoder.variables
            variables=encoder.trainable_variables+decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))
            step = optimizer.iterations.numpy()
            print('step={}'.format(step))


            preds = [process_result(result, label_lang) for result in results]

            ground_truths = [l.numpy().decode() for l in ground_truths]

            acc = compute_accuracy(ground_truths, preds)

            tf.summary.scalar('loss', batch_loss,step=step)
            tf.summary.scalar('accuracy', acc,step=step)
            tf.summary.scalar('lr', learning_rate.numpy(),step=step)
            writer.flush()

            # if batch % 9 == 0:
            if step % 30 == 0:
                lr = max(0.00001, start * math.pow(0.99, step//30))
                learning_rate.assign(lr)
                print('Epoch {} Batch {}/{} Loss {:.4f}  acc {:f}'.format(epoch + 1, batch, N_BATCH,
                                                                          batch_loss.numpy(),
                                                                          acc))
                path = manager.save(checkpoint_number=step)
                print("model saved to %s" % path)
            if step % 30 == 0:
                for i in range(3):
                    print("real:{:s}  pred:{:s} acc:{:f}".format(ground_truths[i], preds[i],
                                                                 compute_accuracy([ground_truths[i]], [preds[i]])))

                # checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
'''
abels tf.Tensor([ 1 55 14 22 27 33 15 31 34 32 21 18 32  3  2], shape=(15,), dtype=int32) tf.Tensor(b'Paintbrushes', shape=(), dtype=string)
labels tf.Tensor([ 1 42 31 18 14 33 22 28 27 22 32 26 32  3  2], shape=(15,), dtype=int32) tf.Tensor(b'Creationisms', shape=(), dtype=string)
labels tf.Tensor([ 1 57 18 22 26 15 34 31 32 22 27 20  3  2  0], shape=(15,), dtype=int32) tf.Tensor(b'Reimbursing', shape=(), dtype=string)
Ti
'''