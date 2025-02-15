from keras import layers, models
import tensorflow as tf

class NetworkBuilder:
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()

    def build_actor_model(self):
        actor_input = layers.Input(shape=self.input_shape, dtype=tf.float32)
        x = layers.Flatten()(actor_input)
        x = layers.Dense(512, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(64, activation='elu')(x)
        x = layers.Dropout(0.5)(x)
        actor_output = layers.Dense(self.action_space, activation='softmax')(x)
        actor_model = models.Model(inputs=actor_input, outputs=actor_output)
        return actor_model

    def build_critic_model(self):
        critic_input = layers.Input(shape=self.input_shape, dtype=tf.float32)
        x = layers.Flatten()(critic_input)
        x = layers.Dense(512, activation='elu')(x)
        x = layers.Dense(256, activation='elu')(x)
        x = layers.Dense(64, activation='elu')(x)
        x = layers.Dropout(0.5)(x)
        critic_output = layers.Dense(1, activation=None)(x)
        critic_model = models.Model(inputs=critic_input, outputs=critic_output)
        return critic_model
