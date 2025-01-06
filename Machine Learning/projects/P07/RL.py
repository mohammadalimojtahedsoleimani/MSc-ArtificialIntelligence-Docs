import random


class TrashCollectionRobot:
    def __init__(self):
        self.state = 0
        self.battery = 100
        self.garbage = 3
        self.garbage_exist = True

    def move(self, destination):
        if self.battery > 0:
            if abs(destination - self.state) == 1 and 0 <= destination <= 5:
                self.state = destination
                self.battery -= 10
                print(f"Moved to state s{self.state}")
            elif destination - self.state == 0 and self.state != 0:
                self.battery -= 5
                print("Bot stayed in previous state!")
        else:
            print("cant move battery is empty")

    def charge(self):
        if self.state == 0:
            self.battery = 100
            print("Battery fully charged.")

    def collect_trash(self):
        if self.garbage_exist and self.state == self.garbage:
            self.garbage_exist = False
            self.garbage = -99
            print(f"Trash collected from state s{self.state}")
        else:
            print("No trash to collect in this state.")

    def sense(self):
        trash_status = "Exists" if self.garbage_exist else "None"
        return {
            "previous state": f"s{self.state - 1 if self.state > 0 else self.state}",
            "current state": f"s{self.state}",
            "Battery": f"{self.battery}%",
            "Garbage": trash_status
        }

    def decide_action(self):
        if self.battery <= 50 and self.state != 0:
            # If battery is low and not in charging state, move towards charging state
            if self.state > 0:
                return "move", 0  # Move to s0
        elif self.garbage_exist and self.state == self.garbage:
            return "collect_trash", None
        else:
            next_state = random.choice([self.state - 1,self.state, self.state + 1])
            if next_state > 5:
                next_state = 5
            if next_state < 0:
                next_state = 0
            return "move", next_state

    def run_mdp(self, steps):
        for _ in range(steps):
            print("\n--- Current State Information ---")
            print(self.sense())

            action, target = self.decide_action()

            if action == "move":
                self.move(target)
            elif action == "collect_trash":
                self.collect_trash()
            else:
                self.charge()

            if action != "move":
                if self.battery > 0:
                    self.battery -= 5
                    print("(Battery level reduce 5% due to no movement)")
            elif action != "move" and self.state == 0:
                self.charge()
                print("Battery fully charged.")

            if self.battery <= 0 and self.state != 0:
                print("Battery depleted! Robot stranded.")
                break



robot = TrashCollectionRobot()
robot.run_mdp(6)
