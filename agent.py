import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

epsilon = 0  # randomness

gameMode = "train"

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Control randomness
        self.gamma = 0.9  # Discount rate; controls tradeoff between short-term and long-term reward: 1 = short-term, 0 = long-term
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when full
        self.model = Linear_QNet(
            11, 256, 3
        )  # 11 since there are exactly 11 state values, 256 hidden layer, 3 since the action has 3 components
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Check for danger in each direction of the head (possible collisions within one block)
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Find current direction, current direction returns True, rest False
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction; Only one can be True
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # Food is left of the head
            game.food.x > game.head.x,  # Food is right of the head
            game.food.y < game.head.y,  # Food is above the head
            game.food.y > game.head.y,  # Food is below the head
        ]
        return np.array(state, dtype=int)  # Converts True/False to 1/0

    def remember(self, state, action, reward, next_state, done):
        # Adds last 2 moves/outcomes to memory
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # If MAX_MEMORY is reached, oldest memory is removed (popleft)

    def train_long_memory(self):
        if (
            len(self.memory) > BATCH_SIZE
        ):  # If memory has at least batch_size number of moves, take a random batch_sized sample of moves
            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )  # Samples BATCH_SIZE number of random moves from self.memory
        else:
            mini_sample = (
                self.memory
            )  # If memory has less than batch_size number of moves, sample all moves

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )  # Unpacks mini_sample into 5 lists; all states together, all actions together, etc.
        self.trainer.train_step(
            states, actions, rewards, next_states, dones
        )  # Train on random batch_sized sample of moves

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            state, action, reward, next_state, done
        )  # Train on last move

    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        # More random moves at the beginning(exploration), less as the game progresses; since we want to learn from experience(exploitation)
        self.epsilon = (
            epsilon - self.n_games
        )  # Epsilon decreases as the game progresses; Randomness decreases as the game progresses
        final_move = [0, 0, 0]
        if (
            random.randint(0, 200) < self.epsilon
        ):  # Generates a random move if the randint is less than epsilon
            # As epsilon decreases, the probability of a random move decreases
            move = random.randint(0, 2)  # Random move (0, 1, 2)
        else:
            state0 = torch.tensor(
                state, dtype=torch.float
            )  # Convert state to tensor (state is current game info; danger zones, food location, heading direction)
            prediction = self.model(state0)
            # Get the index of the highest value in the prediction tensor; highest value means it is most confident that the move is the best move
            move = torch.argmax(prediction).item()  # item converts tensor to int

        final_move[move] = 1  # Assigns 1 to the direction of the calculated move

        return final_move

    def get_play(self, state):
        self.model.eval()  # Set model to evaluation mode
        action = [0, 0, 0]  # Initialize action
        state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
        prediction = self.model.forward(state0)  # Get prediction
        move = torch.argmax(prediction).item()  # Get move
        action[move] = 1  # Assign 1 to the direction of the calculated move

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 89
    agent = Agent()
    game = SnakeGameAI()
    agent.model.load()
    while True:
        # Get old state of game: reward, game-over-state, score
        state_old = agent.get_state(game)

        # Generate new move based on current state
        final_move = agent.get_action(state_old)

        # Perform move and get new game state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory: based on last and current move
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember: store into deck
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory/experience replay: trains on all previous moves and games played
            # Plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()  # Save model when new high score is reached

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)  # List of all scores
            total_score += score  # Total score of all games
            mean_score = total_score / agent.n_games  # Mean score of all games
            plot_mean_scores.append(mean_score)  # List of all mean scores
            plot(
                plot_scores, plot_mean_scores
            )  # Plot scores and mean scores; plot_scores as x-axis, plot_mean_scores as y-axis


def play():
    agent = Agent()  # Initialize agent
    agent.model.load()  # Load model
    game = SnakeGameAI()  # Initialize game
    while True:  # Play game
        state_old = agent.get_state(game)  # Get state of game
        final_move = agent.get_play(state_old)  # Get move
        done, score = game.play_step(final_move)  # Perform move
        if done:  # If game is over, end loop
            break
    print("Score:", score)  # Print score


if __name__ == "__main__":
    if gameMode == "train":
        train()
    else:
        play()
