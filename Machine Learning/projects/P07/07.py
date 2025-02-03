import random


class TrashCollectionRobot:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.state = 0
        self.battery = 100
        self.garbage_location = 3
        self.garbage_collected = False
        self.total_reward = 0

        # Q-learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Possible actions: 0 (left), 1 (stay), 2 (right)
        self.q_table = {}
        for s in range(6):  # States 0 to 5
            for a in range(3):  # 3 possible actions
                self.q_table[(s, a)] = 0

    def move(self, destination):
        if self.battery > 0:
            if 0 <= destination <= 5:
                if destination != self.state:
                    self.battery -= 10
                    print(f"Moved to state s{destination}")

                elif destination == self.state and self.state != 0:
                    self.battery -= 5
                    print(f"stayed in state s{destination}")

                if destination == 0:
                    self.battery = 100
                    print("Battery fully charged.")

                self.state = destination
            else:
                print("invalid move")
        else:
            print("cant move battery is empty")

    def get_reward(self):

        if self.state == self.garbage_location and not self.garbage_collected:
            return 5

        elif self.state == 0:
            return 1
        else:
            return 0

    def choose_action(self):

        # Prioritize moving towards garbage if it exists and battery is > 50%
        if not self.garbage_collected and self.battery > 50:
            if self.state < self.garbage_location:
                action = 2  # Move right
                next_state = self.state + 1
            elif self.state > self.garbage_location:
                action = 0  # Move left
                next_state = self.state - 1
            else:  # Already at garbage location
                action = 1  # Stay
                next_state = self.state

            return action, next_state


        if self.battery <= 50:
            if self.state > 0:
                action = 0  # Move left
                next_state = self.state - 1
            elif self.state == 0:
                action = 1
                next_state = self.state
            else:
                action = 2
                next_state = self.state + 1

            return action, next_state


        if random.random() < self.epsilon:  # Explore
            action = random.choice([0, 1, 2])  # 0: left, 1: stay, 2: right
        else:  # Exploit
            q_values = [self.q_table[(self.state, a)] for a in range(3)]
            max_q = max(q_values)
            actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q]
            action = random.choice(actions_with_max_q)


        if action == 0:  # Move left
            next_state = max(0, self.state - 1)
        elif action == 1:  # Stay
            next_state = self.state
        else:  # Move right
            next_state = min(5, self.state + 1)

        return action, next_state

    def update_q_table(self, action, next_state, reward):

        current_q = self.q_table[(self.state, action)]
        max_next_q = max([self.q_table[(next_state, a)] for a in range(3)])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(self.state, action)] = new_q

    def sense(self):
        trash_status = "Exists" if not self.garbage_collected else "None"
        return {
            "current state": f"s{self.state}",
            "Battery": f"{self.battery}%",
            "Garbage": trash_status,
            "Total Reward": self.total_reward,
            "Garbage location": f"s{self.garbage_location}"
        }

    def run_mdp(self, steps):
        print("\n--- Initial State Information ---")
        print(self.sense())

        for _ in range(steps):
            action, next_state = self.choose_action()


            temp_state = self.state
            self.state = next_state
            reward = self.get_reward()
            self.state = temp_state

            if self.battery - (10 if next_state != self.state else 5) <= 0 and next_state != 0:
                print("Prevented move to avoid battery depletion below 0")
                # self.battery = 0


                next_state = 0
                action = 0
                reward = self.get_reward()
                self.move(next_state)
                self.total_reward += reward
                self.update_q_table(action, next_state, reward)

                if self.state == self.garbage_location and not self.garbage_collected:
                    print(f"Trash collected from state s{self.state}")
                    self.garbage_collected = True

                print("\n--- Current State Information ---")
                print(self.sense())
                continue

            self.move(next_state)
            self.total_reward += reward

            if self.state == self.garbage_location and not self.garbage_collected:
                print(f"Trash collected from state s{self.state}")
                self.garbage_collected = True

            self.update_q_table(action, next_state, reward)


            if self.battery <= 0:
                print("prevent battery depletion below 0")

            print("\n--- Current State Information ---")
            print(self.sense())

        print("\n--- Q-table ---")
        for s in range(6):
            for a in range(3):
                print(f"Q(s{s}, a{a}): {self.q_table[(s, a)]:.2f}")



robot = TrashCollectionRobot()
robot.run_mdp(10)