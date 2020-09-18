import sys
sys.path.append('../..')

#%% md

### Import Dependencies

# - [pandas](https://pandas.pydata.org/) - library that we will use for loading and displaying the data in a table
# - [numpy](http://www.numpy.org/) - library that we will use for linear algebra operations
# - [tensorflow](https://www.tensorflow.org/) - library that we will use for training the model
# - [matplotlib](https://matplotlib.org/) - library that we will use for plotting the data

#%%

import numpy as np
import tensorflow as tf

from src.util import timethis
from src.recommender.Evaluator import Evaluator

np.random.seed(25092020)

#%% md

### Read The Dataset


#%%

from src.dataset.dataset import DataLoader

data = DataLoader(path_train_data='./data/movielens-500/trainingset.tsv'
                      , path_test_data='./data/movielens-500/testset.tsv')

#%% md

#### Print Some Statistics on the Dataset

#%% md

### Define The Model

#%%

from src.recommender.RecommenderModel import RecommenderModel

class BPRMF(RecommenderModel):
    def __init__(self, data_loader, path_output_rec_result, path_output_rec_weight):
        super(BPRMF, self).__init__(data_loader, path_output_rec_result, path_output_rec_weight, 'bprmf')
        self.embedding_size = 64
        self.learning_rate = 0.05
        self.reg = 0
        self.epochs = 5
        self.batch_size = 512
        self.verbose = 1
        self.evaluator = Evaluator(self, data, 100) # Evaluates on TOP-100 Recommendation Lists

        self.initialize_model_parameters()
        self.initialize_perturbations()
        self.initialize_optimizer()

    def initialize_model_parameters(self):
        """
            Initialize Model Parameters
        """
        self.embedding_P = tf.Variable(tf.random.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01))  # (users, embedding_size)
        self.embedding_Q = tf.Variable(tf.random.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01))  # (items, embedding_size)
        self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1])

    def initialize_optimizer(self):
        """
            Optimizer
        """
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)

    def initialize_perturbations(self):
        """
            Set delta variables useful to store delta perturbations,
        """
        self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]), trainable=False)
        self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]), trainable=False)

    def get_inference(self, user_input, item_input_pos):
        """
            Generate Prediction Matrix with respect to passed users and items identifiers
        """
        self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P + self.delta_P, user_input), 1)
        self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q + self.delta_Q, item_input_pos), 1)

        return tf.matmul(self.embedding_p * self.embedding_q,self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    def get_full_inference(self):
        """
            Get Full Predictions useful for Full Store of Predictions
        """
        return tf.matmul(self.embedding_P + self.delta_P, tf.transpose(self.embedding_Q + self.delta_Q))

    @timethis
    def _train_step(self, batches):
        """
            Apply a Single Training Step (across all the batches in the dataset).
        """
        user_input, item_input_pos, item_input_neg = batches

        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                # Model Inference
                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)

                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

                # Loss Function
                self.loss_opt = self.loss + self.reg_loss

            gradients = t.gradient(self.loss_opt, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))

    @timethis
    def train(self):
        for epoch in range(self.epochs):
            batches = self.data.shuffle(self.batch_size)
            self._train_step(batches)
            print('Epoch {0}/{1}'.format(epoch+1, self.epochs))

    @timethis
    def _adversarial_train_step(self, batches, epsilon):
        """
            Apply a Single Training Step (across all the batches in the dataset).
        """
        user_input, item_input_pos, item_input_neg = batches
        adv_reg = 1
        self.initialize_perturbations()
        print(len(user_input))
        for batch_idx in range(len(user_input)):
            with tf.GradientTape() as t:
                t.watch([self.embedding_P, self.embedding_Q])

                # Model Inference
                self.output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[batch_idx],
                                                                               item_input_pos[batch_idx])
                self.output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[batch_idx],
                                                                               item_input_neg[batch_idx])
                self.result = tf.clip_by_value(self.output_pos - self.output_neg, -80.0, 1e8)

                self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

                # Regularization Component
                self.reg_loss = self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

                # Adversarial Regularization Component
                ##  Execute the Adversarial Attack on the Current Model (Perturb Model Parameters)
                self.execute_adversarial_attack(epsilon)
                ##  Inference on the Adversarial Perturbed Model
                self.output_pos_adver, _, _ = self.get_inference(user_input[batch_idx], item_input_pos[batch_idx])
                self.output_neg_adver, _, _ = self.get_inference(user_input[batch_idx], item_input_neg[batch_idx])

                self.result_adver = tf.clip_by_value(self.output_pos_adver - self.output_neg_adver, -80.0, 1e8)
                self.loss_adver = tf.reduce_sum(tf.nn.softplus(-self.result_adver))

                # Loss Function
                self.adversarial_regularizer = adv_reg * self.loss_adver # AMF = Adversarial Matrix Factorization
                self.bprmf_loss = self.loss + self.reg_loss

                self.amf_loss = self.bprmf_loss + self.adversarial_regularizer

            gradients = t.gradient(self.amf_loss, [self.embedding_P, self.embedding_Q])
            self.optimizer.apply_gradients(zip(gradients, [self.embedding_P, self.embedding_Q]))

        self.initialize_perturbations()


    @timethis
    def adversarial_train(self, adversarial_epochs, epsilon):
        for epoch in range(adversarial_epochs):
            batches = self.data.shuffle(self.batch_size)
            self._adversarial_train_step(batches, epsilon)
            print('Epoch {0}/{1}'.format(self.epochs+epoch+1, self.epochs+adversarial_epochs))

    def execute_adversarial_attack(self, epsilon):
        user_input, item_input_pos, item_input_neg = self.data.shuffle(len(self.data._user_input))

        with tf.GradientTape() as tape_adv:
            tape_adv.watch([self.embedding_P, self.embedding_Q])
            # Evaluate Current Model Inference
            output_pos, embed_p_pos, embed_q_pos = self.get_inference(user_input[0],
                                                                      item_input_pos[0])
            output_neg, embed_p_neg, embed_q_neg = self.get_inference(user_input[0],
                                                                      item_input_neg[0])
            result = tf.clip_by_value(output_pos - output_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))
            loss += self.reg * tf.reduce_mean(
                tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))
        # Evaluate the Gradient
        grad_P, grad_Q = tape_adv.gradient(loss, [self.embedding_P, self.embedding_Q])
        grad_P, grad_Q = tf.stop_gradient(grad_P), tf.stop_gradient(grad_Q)

        # Use the Gradient to Build the Adversarial Perturbations (https://doi.org/10.1145/3209978.3209981)
        self.delta_P = tf.nn.l2_normalize(grad_P, 1) * epsilon
        self.delta_Q = tf.nn.l2_normalize(grad_Q, 1) * epsilon




#%% md

### Initialize and Train The Model


#%%

recommender_model = BPRMF(data, '../rec_result/', '../rec_weights/')

recommender_model.train()

#%% md

### Evaluated The Model

#%%

before_adv_hr, before_adv_ndcg, before_adv_auc = recommender_model.evaluator.evaluate()


#%% md

### Adversarial Attack Against The Model

#%%

recommender_model.execute_adversarial_attack(epsilon=0.1)

#%% md

### Evaluate the Effects of the Adversarial Attack

#%%

after_adv_hr, after_adv_ndcg, after_adv_auc = recommender_model.evaluator.evaluate()

print('HR decreases by %.2f%%' % ((1-after_adv_hr/before_adv_hr)*100))
print('nDCG decreases by %.2f%%' % ((1-after_adv_ndcg/before_adv_ndcg)*100))
print('AUC decreases by %.2f%%' % ((1-after_adv_auc/before_adv_auc)*100))

#%% md

### Implement The Adversarial Training/Regularization

#%%

recommender_model.adversarial_train(adversarial_epochs=5, epsilon=0.1)


#%% md

### Evaluated The Adversarial Defended Model before the Attack

#%%

before_adv_hr, before_adv_ndcg, before_adv_auc = recommender_model.evaluator.evaluate()


#%% md

### Adversarial Attack Against The Defended Model

#%%

recommender_model.execute_adversarial_attack(epsilon=0.1)

#%% md

### Evaluate the Effects of the Adversarial Attack against the Defended Model

#%%

after_adv_hr, after_adv_ndcg, after_adv_auc = recommender_model.evaluator.evaluate()

print('HR decreases by %.2f%%' % ((1-after_adv_hr/before_adv_hr)*100))
print('nDCG decreases by %.2f%%' % ((1-after_adv_ndcg/before_adv_ndcg)*100))
print('AUC decreases by %.2f%%' % ((1-after_adv_auc/before_adv_auc)*100))
