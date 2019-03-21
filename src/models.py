import tensorflow as tf

VOCAB_SIZE = 12602
MAX_QUESTION_LEN = 26


class ShowNTellNet:
    def __init__(self, question_embed_dim, lstm_dim, n_answers):
        self.question_embed_dim = question_embed_dim
        self.lstm_dim = lstm_dim
        self.n_answers = n_answers
        self.model = None
        self.build()

    def build(self):
        image_features = tf.keras.layers.Input(shape=(4096,),
                                               dtype='float32')

        image_embedding = tf.keras.layers.Dense(units=self.question_embed_dim,
                                                activation='elu',
                                                name='image_embedding')(inputs=image_features)

        image_embedding = tf.keras.layers.Reshape((1, self.question_embed_dim))(image_embedding)

        question_input = tf.keras.layers.Input(shape=(MAX_QUESTION_LEN,),
                                               dtype='int32',
                                               name='question_input')

        question_input_masked = tf.keras.layers.Masking(mask_value=0,
                                                        input_shape=(MAX_QUESTION_LEN,),
                                                        name='question_input_masked')(inputs=question_input)

        question_embedding = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE,
                                                       output_dim=self.question_embed_dim,
                                                       input_length=MAX_QUESTION_LEN,
                                                       name='question_embedding')(inputs=question_input_masked)

        image_question_embedding = tf.keras.layers.Concatenate(axis=1,
                                                               name='image_question_embedding')(inputs=[image_embedding, question_embedding])

        question_features, last_h, _ = tf.keras.layers.LSTM(units=self.lstm_dim,
                                                            return_sequences=True,
                                                            return_state=True,
                                                            name='question_generator')(inputs=image_question_embedding)

        question_pred = tf.keras.layers.TimeDistributed(layer=tf.keras.layers.Dense(units=VOCAB_SIZE,
                                                                                    activation='softmax'),
                                                        name='word_classify')(inputs=question_features)

        # question_pred[:-1] ignores the last output. Need to add <START> and <END>.
        question_pred = tf.keras.layers.Lambda(lambda x: x[:, :-1, :],
                                               name='word_classifier')(inputs=question_pred)

        answer_fc_1 = tf.keras.layers.Dense(units=1000,
                                           activation='elu',
                                           name='answer_fc_1')(inputs=last_h)

        answer_fc_2 = tf.keras.layers.Dense(units=1000,
                                           activation='elu',
                                           name='answer_fc_2')(inputs=answer_fc_1)

        answer_pred = tf.keras.layers.Dense(units=self.n_answers,
                                            activation='softmax',
                                            name='answer_classifier')(inputs=answer_fc_2)

        self.model = tf.keras.Model(inputs=[image_features, question_input],
                                    outputs=[question_pred, answer_pred])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_split=0.2,
                       shuffle=True)


class VQANet:
    def __init__(self, question_embed_dim, lstm_dim, n_answers):
        self.question_embed_dim = question_embed_dim
        self.lstm_dim = lstm_dim
        self.n_answers = n_answers
        self.model = None
        self.build()

    def build(self):
        image_features = tf.keras.layers.Input(shape=(4096,),
                                               dtype='float32')

        image_embedding = tf.keras.layers.Dense(units=self.question_embed_dim,
                                                activation='elu',
                                                name='image_embedding')(inputs=image_features)

        image_embedding = tf.keras.layers.RepeatVector(MAX_QUESTION_LEN)(inputs=image_embedding)

        question_input = tf.keras.layers.Input(shape=(MAX_QUESTION_LEN,),
                                               dtype='int32',
                                               name='question_input')

        question_input_masked = tf.keras.layers.Masking(mask_value=0,
                                                        input_shape=(MAX_QUESTION_LEN,),
                                                        name='question_input_masked')(inputs=question_input)

        question_embedding = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE,
                                                       output_dim=self.question_embed_dim,
                                                       input_length=MAX_QUESTION_LEN,
                                                       name='question_embedding')(inputs=question_input_masked)

        image_question_embedding = tf.keras.layers.Concatenate(axis=-1,
                                                               name='image_question_embedding')(inputs=[image_embedding, question_embedding])

        question_features, last_h, _ = tf.keras.layers.LSTM(units=self.lstm_dim,
                                                            return_sequences=True,
                                                            return_state=True,
                                                            name='question_generator')(inputs=image_question_embedding)

        question_pred = tf.keras.layers.TimeDistributed(layer=tf.keras.layers.Dense(units=VOCAB_SIZE,
                                                                                    activation='softmax'),
                                                        name='word_classifier')(inputs=question_features)

        answer_fc_1 = tf.keras.layers.Dense(units=1000,
                                            activation='elu',
                                            name='answer_fc_1')(inputs=last_h)

        answer_fc_2 = tf.keras.layers.Dense(units=1000,
                                            activation='elu',
                                            name='answer_fc_2')(inputs=answer_fc_1)

        answer_pred = tf.keras.layers.Dense(units=self.n_answers,
                                            activation='softmax',
                                            name='answer_classifier')(inputs=answer_fc_2)

        self.model = tf.keras.Model(inputs=[image_features, question_input],
                                    outputs=[question_pred, answer_pred])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_split=0.2,
                       shuffle=True)