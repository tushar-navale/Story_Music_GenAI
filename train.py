import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from melodygenerator import MelodyGenerator
from melodypreprocessor import MelodyPreprocessor
from transformer import Transformer

# Global parameters
EPOCHS = 10
BATCH_SIZE = 32
DATA_PATH = "dataset.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 100

# Loss function and optimizer
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
optimizer = Adam()


def train(train_dataset, transformer, epochs):
    """
    Trains the Transformer model on a given dataset for a specified number of epochs.
    """
    print("Training the model...")
    for epoch in range(epochs):
        total_loss = 0
        for (batch, (input, target)) in enumerate(train_dataset):
            batch_loss = _train_step(input, target, transformer)
            total_loss += batch_loss
            print(f"Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy():.4f}")


@tf.function
def _train_step(input, target, transformer):
    """
    Performs a single training step for the Transformer model.
    """
    # Prepare input for the decoder
    target_input = _right_pad_sequence_once(target[:, :-1])
    target_real = _right_pad_sequence_once(target[:, 1:])

    with tf.GradientTape() as tape:
        # Corrected model call using keyword arguments for input and target
        predictions = transformer(
            inputs=(input, target_input),
            
            training=True,
            enc_padding_mask=None,
            look_ahead_mask=None,
            dec_padding_mask=None,
        )

        loss = _calculate_loss(target_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss


def _calculate_loss(real, pred):
    """
    Computes the loss between the real and predicted sequences.
    """
    loss_ = sparse_categorical_crossentropy(real, pred)

    # Mask padded values (assumed to be 0)
    mask = tf.math.not_equal(real, 0)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    total_loss = tf.reduce_sum(loss_)
    num_non_padding = tf.reduce_sum(mask)
    return total_loss / num_non_padding


def _right_pad_sequence_once(sequence):
    """
    Pads a sequence with a single zero at the end.
    """
    return tf.pad(sequence, [[0, 0], [0, 1]], "CONSTANT")


if __name__ == "__main__":
    melody_preprocessor = MelodyPreprocessor(DATA_PATH, batch_size=BATCH_SIZE)
    train_dataset = melody_preprocessor.create_training_dataset()
    vocab_size = melody_preprocessor.number_of_tokens_with_padding

    transformer_model = Transformer(
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_feedforward=128,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_num_positions_in_pe_encoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        max_num_positions_in_pe_decoder=MAX_POSITIONS_IN_POSITIONAL_ENCODING,
        dropout_rate=0.1,
    )

    train(train_dataset, transformer_model, EPOCHS)

    print("Generating a melody...")
    melody_generator = MelodyGenerator(
        transformer_model, melody_preprocessor.tokenizer
    )
    start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
    new_melody = melody_generator.generate(start_sequence)
    print(f"Generated melody: {new_melody}")
