import os
import time
import math
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from gym_torcs import TorcsEnv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from ReplayBuffer import ReplayBuffer
from OU import OU as OrnsteinUhlenbeck

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Neural network configuration
HIDDEN_LAYER1 = 64
HIDDEN_LAYER2 = 32

def construct_actor(input_dim):
    """Build the actor network for policy approximation."""
    state_input = Input(shape=(input_dim,))
    x = Dense(HIDDEN_LAYER1, activation='relu')(state_input)
    x = BatchNormalization()(x)
    x = Dense(HIDDEN_LAYER2, activation='relu')(x)
    x = BatchNormalization()(x)

    steer_out = Dense(1, activation='tanh')(x)
    accel_out = Dense(1, activation='sigmoid')(x)
    brake_out = Dense(1, activation='sigmoid')(x)

    final_output = concatenate([steer_out, accel_out, brake_out])
    return Model(inputs=state_input, outputs=final_output)

def construct_critic(state_size, action_size):
    """Build the critic network for Q-value estimation."""
    state_input = Input(shape=(state_size,))
    action_input = Input(shape=(action_size,), name='action_input')

    state_branch = Dense(HIDDEN_LAYER1, activation='relu')(state_input)
    state_branch = BatchNormalization()(state_branch)
    state_branch = Dense(HIDDEN_LAYER2, activation='linear')(state_branch)
    action_branch = Dense(HIDDEN_LAYER2, activation='linear')(action_input)

    combined = concatenate([state_branch, action_branch])
    combined_out = Dense(HIDDEN_LAYER2, activation='relu')(combined)
    combined_out = BatchNormalization()(combined_out)
    q_value_output = Dense(1, activation='linear')(combined_out)

    model = Model(inputs=[state_input, action_input], outputs=q_value_output)
    model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')  # Increased critic LR
    return model

def execute_training(train_mode=1):
    """Run the DDPG training loop."""
    buffer_capacity = 100000
    batch_size = 32
    gamma = 0.99
    tau = 0.001
    actor_learning_rate = 0.0003  # Increased for faster learning
    critic_learning_rate = 0.002

    state_size = 30  # Increased to include track curvature
    action_size = 3
    episodes = 2000
    max_steps = 100
    decay_factor = 100000

    visual_input = False
    exploration_factor = 1.0
    noise_generator = OrnsteinUhlenbeck()

    replay_memory = ReplayBuffer(buffer_capacity)
    actor_eval = construct_actor(state_size)
    actor_target = construct_actor(state_size)
    critic_eval = construct_critic(state_size, action_size)
    critic_target = construct_critic(state_size, action_size)

    environment = TorcsEnv(vision=visual_input, throttle=True, gear_change=False)

    print("Checking for existing models...")
    try:
        actor_eval.load_weights("actormodel.h5")
        actor_target.load_weights("actormodel.h5")
        critic_eval.load_weights("criticmodel.h5")
        critic_target.load_weights("criticmodel.h5")
        print("Model weights loaded.")
    except:
        print("No existing models found.")

    print("Starting training loop in TORCS...")

    global_step = 0
    last_action = np.zeros(action_size)  # For temporal smoothing

    for episode in range(episodes):
        print(f"Episode {episode} | Replay size: {replay_memory.count()}")

        if episode % 3 == 0:
            obs = environment.reset(relaunch=True)
        else:
            obs = environment.reset()

        # Compute track curvature (difference in track sensors)
        track = np.array(obs.track)
        curvature = np.std(track[:9] - track[10:]) / 200.0  # Normalized
        state = np.hstack((
            obs.angle, obs.track, obs.trackPos,
            obs.speedX, obs.speedY, obs.speedZ,
            obs.wheelSpinVel / 100.0, obs.rpm,
            curvature
        ))

        total_reward = 0.0
        warmup_steps = 10
        smoothing_alpha = 0.5  # Reduced for responsiveness

        for step in range(max_steps):
            cpu_start_time = time.process_time()

            exploration_factor = max(0, exploration_factor - 1.0 / decay_factor)
            current_action = np.zeros((1, action_size))
            added_noise = np.zeros((1, action_size))

            # Check for stuck condition
            is_stuck = (state[4] < 5 and min(state[5:24]) < 0) or min(state[25:61]) < 10 / 200.0  # Low speed, off-track, or opponent close

            if step < warmup_steps:
                current_action[0][0] = 0.0
                current_action[0][1] = 0.5
                current_action[0][2] = 0.0
                print(f"Warmup Step {step}: Forcing action: steer={current_action[0][0]:.3f}, accel={current_action[0][1]:.3f}, brake={current_action[0][2]:.3f}")
            elif is_stuck:
                current_action[0][0] = random.uniform(-1, 1)  # Random steering to escape
                current_action[0][1] = 0.0  # No acceleration
                current_action[0][2] = 0.5  # Apply brake to simulate reverse
                print(f"Stuck Recovery: steer={current_action[0][0]:.3f}, accel={current_action[0][1]:.3f}, brake={current_action[0][2]:.3f}, opponent_min={min(state[25:61]):.3f}")
            else:
                chosen_action = actor_eval.predict(state.reshape(1, -1))
                # Adjusted noise parameters
                added_noise[0][0] = train_mode * exploration_factor * noise_generator.function(chosen_action[0][0], 0.0, 0.6, 0.4)
                added_noise[0][1] = train_mode * exploration_factor * noise_generator.function(chosen_action[0][1], 0.5, 0.5, 0.03)
                added_noise[0][2] = train_mode * exploration_factor * noise_generator.function(chosen_action[0][2], 0.0, 0.3, 0.02)

                current_action[0] = chosen_action[0] + added_noise[0]
                # Apply temporal smoothing
                current_action[0] = smoothing_alpha * current_action[0] + (1 - smoothing_alpha) * last_action
                last_action = current_action[0].copy()

                current_action[0][0] = np.clip(current_action[0][0], -1.0, 1.0)
                current_action[0][1] = np.clip(current_action[0][1], 0.0, 1.0)
                current_action[0][2] = np.clip(current_action[0][2], 0.0, 1.0)

                print(f"Step {step}: Predicted: {chosen_action[0]}, Noise: {added_noise[0]}, Smoothed: {current_action[0]}")

            new_obs, reward, done, _ = environment.step(current_action[0])

            new_track = np.array(new_obs.track)
            new_curvature = np.std(new_track[:9] - new_track[10:]) / 200.0
            new_state = np.hstack((
                new_obs.angle, new_obs.track, new_obs.trackPos,
                new_obs.speedX, new_obs.speedY, new_obs.speedZ,
                new_obs.wheelSpinVel / 100.0, new_obs.rpm,
                new_curvature
            ))

            replay_memory.add(state, current_action[0], reward, new_state, done)

            sampled_data = replay_memory.getBatch(batch_size)
            print(f"Mini-batch size: {len(sampled_data)}")

            state_batch = np.array([x[0] for x in sampled_data])
            action_batch = np.array([x[1] for x in sampled_data])
            reward_batch = np.array([x[2] for x in sampled_data])
            next_state_batch = np.array([x[3] for x in sampled_data])
            done_batch = np.array([x[4] for x in sampled_data])

            next_actions = actor_target.predict(next_state_batch)
            predicted_q = critic_target.predict([next_state_batch, next_actions])

            targets = np.zeros_like(reward_batch)
            for i in range(len(sampled_data)):
                if done_batch[i]:
                    targets[i] = reward_batch[i]
                else:
                    targets[i] = reward_batch[i] + gamma * predicted_q[i]

            if train_mode:
                loss_val = critic_eval.train_on_batch([state_batch, action_batch], targets)

                with tf.GradientTape() as tape:
                    predicted_actions = actor_eval(state_batch)
                    critic_output = critic_eval([state_batch, predicted_actions])
                    actor_loss = -tf.reduce_mean(critic_output)
                actor_grads = tape.gradient(actor_loss, actor_eval.trainable_variables)
                Adam(learning_rate=actor_learning_rate).apply_gradients(zip(actor_grads, actor_eval.trainable_variables))

                for j, (eval_w, target_w) in enumerate(zip(critic_eval.trainable_weights, critic_target.trainable_weights)):
                    target_w.assign(tau * eval_w + (1 - tau) * target_w)
                for j, (eval_w, target_w) in enumerate(zip(actor_eval.trainable_weights, actor_target.trainable_weights)):
                    target_w.assign(tau * eval_w + (1 - tau) * target_w)

                total_reward += reward
                state = new_state
                print(f"Step CPU Time: {time.process_time() - cpu_start_time}")

            print(f"Episode {episode}, Global Step {global_step}, Reward: {reward:.2f}, Loss: {loss_val:.4f}, Done: {done}")
            global_step += 1

            if done:
                print(f"Episode terminated early at step {step}")
                break

        if episode % 3 == 0 and train_mode:
            print("Persisting model weights...")
            actor_eval.save_weights("actormodel.h5", overwrite=True)
            with open("actormodel.json", "w") as f:
                json.dump(actor_eval.to_json(), f)

            critic_eval.save_weights("criticmodel.h5", overwrite=True)
            with open("criticmodel.json", "w") as f:
                json.dump(critic_eval.to_json(), f)

        print(f"Episode {episode} Total Reward: {total_reward:.2f}")
        print(f"Total Steps: {global_step}\n")

    environment.end()
    print("Session complete.")

if __name__ == "__main__":
    execute_training()