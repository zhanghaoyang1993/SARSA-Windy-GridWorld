import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7
# world width
WORLD_WIDTH = 10
# wind strenth for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
# probability for exploration
EPSILON = 0.1
# SARSA step size
ALPHA = 0.5
# reward for each step
REWARD = -1
START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0),
                j]  # 向上是-1，grid world最上面一行的纵坐标为0，最下面一行纵坐标为6，所以向上移动时选max（state，0），即state的纵坐标不能小于0
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]  # 同理，向下移动时，state的纵坐标必须小于6，大于0
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


# play an episode
def episode(q_value):
    # track the total time steps in this episode
    time = 0
    # initialize state
    state = START
    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:  # EPLISION > random(0,1)
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]  # state[0]为横坐标，state[1]为纵坐标，：为actions；q_value[x,y]，x为state，y为actions
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(
            values_)])  # enumerate是枚举，这里表示枚举出value_中的值及其序号，最后选取最大values_所对应的action
    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # SARSA update
        q_value[state[0], state[1], action] += ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def sarsa():
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))  # q_value初始值
    episode_limit = 500
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value))  # 将time存入steps
        #time = episode(q_value)
        #episode.extend([ep] * time)  # 将ep*time存入episode
        ep += 1
    steps = np.add.accumulate(steps)  # 累加序列
    plt.plot(steps, np.arange(0, len(steps + 1)))  # np.arange(1,len(steps+1))生产1到len(steps)的数列
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.savefig('./sarsa.png')
    plt.close()
    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')  # [-1]意思为在list末尾添加值
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':
    sarsa()
