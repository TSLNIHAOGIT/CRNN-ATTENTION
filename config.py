from easydict import EasyDict as edict

cfg = edict()

cfg.CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'_&.!?,\""

cfg.LEARNING_RATE = 0.0001

cfg.LR_DECAY_RATE = 0.95

cfg.EMBEDDING_DIM = 50#256

cfg.UNITS = 128#1024

cfg.TRAIN_BATCH_SIZE = 30

