import random


class TrashCollectionRobot:
    def __init__(self):
        self.state = 0
        self.battery = 100
        self.garbage = 3
        self.garbage_exist = True
        self.reward = 0

    def move(self, destination):
        if self.battery > 0:
            if abs(destination - self.state) == 1 and 0 <= destination <= 5:
                self.state = destination
                self.battery -= 10
                print(f"Moved to state s{self.state}")
                if self.state == self.garbage:
                    self.garbage_exist = False
                    self.garbage = -99
                    print(f"Trash collected from state s{self.state}")
            elif destination - self.state == 0 and self.state != 0:
                self.battery -= 5
                print("Bot stayed in previous state!")
        else:
            print("cant move battery is empty")

    def charge(self):
        if self.state == 0:
            self.battery = 100
            print("Battery fully charged.")

    def sense(self):
        trash_status = "Exists" if self.garbage_exist else "None"
        return {
            "previous state": f"s{self.state - 1 if self.state > 0 else self.state}",
            "current state": f"s{self.state}",
            "Battery": f"{self.battery}%",
            "Garbage": trash_status,
            "Reward": f"{self.reward}"
        }

    def get_reward(self):
        if self.state == self.garbage:
            return 5
        elif self.state == 0:
            return 1
        else:
            return 0

    def decide_action(self):
        if self.battery <= 50 and self.state != 0:
            # If battery is low and not in charging state, move towards charging state
            if self.state > 0:
                return "move", 0  # Move to s0

        possible_actions = []
        if self.state > 0:
            possible_actions.append(("move", self.state - 1))
        if self.state < 5:
            possible_actions.append(("move", self.state + 1))
        possible_actions.append(("move", self.state))
        best_action = None
        best_reward = -float('inf')

        for action, next_state in possible_actions:
            reward = 0
            temp_state = self.state

            if action == "move":
                self.state = next_state
                reward = self.get_reward()
                self.state = temp_state

            if reward > best_reward:
                best_reward = reward
                best_action = (action, next_state)
                self.reward += best_reward

        return best_action

    def run_mdp(self, steps):
        print("\n--- Initial  State Information ---")
        print(self.sense())
        print()
        for _ in range(steps):

            action, target = self.decide_action()

            if action == "move":
                self.move(target)
            else:
                self.charge()

            if self.state == 0:
                self.charge()

            if self.battery <= 0 and self.state != 0:
                print("Battery depleted! Robot stranded.")
                break

            print("\n--- Current State Information ---")
            print(self.sense())
            print()


robot = TrashCollectionRobot()
robot.run_mdp(6)
